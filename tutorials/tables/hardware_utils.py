from snorkel.models import TemporaryImplicitSpan, CandidateSet, AnnotationKey, AnnotationKeySet, Label
from snorkel.matchers import RegexMatchSpan, Union
from snorkel.utils import ProgressBar
from snorkel.loaders import create_or_fetch
from snorkel.lf_helpers import *
from snorkel.candidates import OmniNgrams
from hardware_spaces import OmniNgramsPart
from hardware_matchers import get_matcher
import csv
import codecs
import re
import os
from collections import defaultdict

from itertools import chain

# eeca_matcher = RegexMatchSpan(rgx='([b]{1}[abcdefklnpqruyz]{1}[\swxyz]?[0-9]{3,5}[\s]?[A-Z\/]{0,5}[0-9]?[A-Z]?([-][A-Z0-9]{1,7})?([-][A-Z0-9]{1,2})?)')
# jedec_matcher = RegexMatchSpan(rgx='([123]N\d{3,4}[A-Z]{0,5}[0-9]?[A-Z]?)')
# jis_matcher = RegexMatchSpan(rgx='(2S[abcdefghjkmqrstvz]{1}[\d]{2,4})')
# others_matcher = RegexMatchSpan(rgx='((NSVBC|SMBT|MJ|MJE|MPS|MRF|RCA|TIP|ZTX|ZT|TIS|TIPL|DTC|MMBT|PZT){1}[\d]{2,4}[A-Z]{0,3}([-][A-Z0-9]{0,6})?([-][A-Z0-9]{0,1})?)')
# part_matcher = Union(eeca_matcher, jedec_matcher, jis_matcher, others_matcher)

def load_hardware_doc_part_pairs(filename):
    with open(filename, 'r') as csvfile:
        gold_reader = csv.reader(csvfile)
        gold = set()
        for row in gold_reader:
            (doc, part, attr, val) = row
            gold.add((doc.upper(), part.upper()))
        return gold


def get_gold_parts(filename, docs=None):
    return set(map(lambda x: x[0], get_gold_dict(filename, doc_on=False, part_on=True, val_on=False, docs=docs)))


def get_gold_dict(filename, doc_on=True, part_on=True, val_on=True, attrib=None, docs=None, integerize=False):
    with codecs.open(filename, encoding="utf-8") as csvfile:
        gold_reader = csv.reader(csvfile)
        gold_dict = set()
        for row in gold_reader:
            (doc, part, attr, val) = row
            if docs is None or doc.upper() in docs:
                if attrib and attr != attrib:
                    continue
                if not val:
                    continue
                else:
                    key = []
                    if doc_on:  key.append(doc.upper())
                    if part_on: key.append(part.upper())
                    if val_on:
                        if integerize:
                            key.append(int(float(val)))
                        else:
                            key.append(val.upper())
                    gold_dict.add(tuple(key))
    return gold_dict


def count_hardware_labels(candidates, filename, attrib, attrib_class):
    gold_dict = get_gold_dict(filename, attrib)
    gold_cand = defaultdict(int)
    pb = ProgressBar(len(candidates))
    for i, c in enumerate(candidates):
        pb.bar(i)
        key = ((c[0].parent.document.name).upper(), (c[0].get_span()).upper(), (''.join(c[1].get_span().split())).upper())
        if key in gold_dict:
            gold_cand[key] += 1
    pb.close()
    return gold_cand


def load_hardware_labels(session, label_set_name, annotation_key_name, candidates, filename, attrib):
    gold_dict = get_gold_dict(filename, attrib=attrib)
    candidate_set   = create_or_fetch(session, CandidateSet, label_set_name)
    annotation_key  = create_or_fetch(session, AnnotationKey, annotation_key_name)
    key_set         = create_or_fetch(session, AnnotationKeySet, annotation_key_name)
    if annotation_key not in key_set.keys:
        key_set.append(annotation_key)
    session.commit()

    cand_total = len(candidates)
    print 'Loading', cand_total, 'candidate labels'
    pb = ProgressBar(cand_total)
    for i, c in enumerate(candidates):
        pb.bar(i)
        doc = (c[0].parent.document.name).upper()
        part = (c[0].get_span()).upper()
        val = (''.join(c[1].get_span().split())).upper()
        if (doc, part, val) in gold_dict:
            candidate_set.append(c)
            session.add(Label(key=annotation_key, candidate=c, value=1))
    session.commit()
    pb.close()
    return (candidate_set, annotation_key)


def most_common_document(candidates):
    """Returns the document that produced the most of the passed-in candidates"""
    # Turn CandidateSet into set of tuples
    pb = ProgressBar(len(candidates))
    candidate_count = {}

    for i, c in enumerate(candidates):
        pb.bar(i)
        part = c.get_arguments()[0].get_span()
        doc = c.get_arguments()[0].parent.document.name
        candidate_count[doc] = candidate_count.get(doc, 0) + 1 # count number of occurences of keys
    pb.close()
    max_doc = max(candidate_count, key=candidate_count.get)
    return max_doc


def separate_fns(fn, cfn):
    unfound = set()
    misclassified = set()
    for c in fn:
        (doc, part, attr) = c
        if (doc, part) in cfn:
            unfound.add(c)
        else:
            misclassified.add(c)
    print "%d fns" % len(fn)
    print "%d unfound" % len(unfound)
    print "%d misclassified" % len(misclassified)
    return unfound, misclassified


def entity_confusion_matrix(pred, gold):
    if not isinstance(pred, set):
        pred = set(pred)
    if not isinstance(gold, set):
        gold = set(gold)
    TP = pred.intersection(gold)
    FP = pred.difference(gold)
    FN = gold.difference(pred)
    return (TP, FP, FN)


def entity_level_f1_from_candidates(candidates, gold_file, attribute, corpus=None, 
                              relation=True, parts_by_doc=None):
    """Checks entity-level recall of candidates compared to gold.

    Turns a CandidateSet into a normal set of entity-level tuples
    (doc, part, [attribute_value])
    then compares this to the entity-level tuples found in the gold.

    Example Usage:
        from hardware_utils import entity_level_total_recall
        candidates = # CandidateSet of all candidates you want to consider
        gold_file = os.environ['SNORKELHOME'] + '/tutorials/tables/data/hardware/hardware_gold.csv'
        entity_level_total_recall(candidates, gold_file, 'stg_temp_min')
    """
    docs = [(doc.name).upper() for doc in corpus.documents.all()] if corpus else None
    gold_set = get_gold_dict(gold_file, docs=docs, doc_on=True, part_on=True, val_on=relation, attrib=attribute)
    if len(gold_set) == 0:
        print "Gold set is empty."
        return
    # Turn CandidateSet into set of tuples
    print "Preparing candidates..."
    pb = ProgressBar(len(candidates))
    entity_level_candidates = set()
    for i, c in enumerate(candidates):
        pb.bar(i)
        part = c.get_arguments()[0].get_span()
        doc = c.get_arguments()[0].parent.document.name.upper()
        if relation:
            val = c.get_arguments()[1].get_span()
        for p in get_implied_parts(part, doc, parts_by_doc):
            if relation:
                entity_level_candidates.add((doc, p, val))
            else:
                entity_level_candidates.add((doc, p))
    pb.close()

    (TP_set, FP_set, FN_set) = entity_confusion_matrix(entity_level_candidates, gold_set)
    TP = len(TP_set)
    FP = len(FP_set)
    FN = len(FN_set)

    prec = TP / float(TP + FP) if TP + FP > 0 else float('nan')
    rec  = TP / float(TP + FN) if TP + FN > 0 else float('nan')
    f1   = 2 * (prec * rec) / (prec + rec) if prec + rec > 0 else float('nan')
    print "========================================"
    print "Scoring on Entity-Level Gold Data"
    print "========================================"
    print "Corpus Precision {:.3}".format(prec)
    print "Corpus Recall    {:.3}".format(rec)
    print "Corpus F1        {:.3}".format(f1)
    print "----------------------------------------"
    print "TP: {} | FP: {} | FN: {}".format(TP, FP, FN)
    print "========================================\n"
    return map(lambda x: sorted(list(x)), [TP_set, FP_set, FN_set])


def entity_level_f1(tp, fp, tn, fn, gold_file, corpus, attrib):
    docs = [(doc.name).upper() for doc in corpus.documents.all()] if corpus else None
    gold_dict = get_gold_dict(gold_file, docs=docs, doc_on=True, part_on=(attrib is not None), val_on=True, attrib=attrib)

    TP = FP = TN = FN = 0
    pos = set([((c[0].parent.document.name).upper(),
                (c[0].get_span()).upper(),
                (c[1].get_span()).upper()) for c in tp.union(fp)])
    TP_set = pos.intersection(gold_dict)
    TP = len(TP_set)
    FP_set = pos.difference(gold_dict)
    FP = len(FP_set)
    FN_set = gold_dict.difference(pos)
    FN = len(FN_set)

    prec = TP / float(TP + FP) if TP + FP > 0 else float('nan')
    rec  = TP / float(TP + FN) if TP + FN > 0 else float('nan')
    f1   = 2 * (prec * rec) / (prec + rec) if prec + rec > 0 else float('nan')
    print "========================================"
    print "Scoring on Entity-Level Gold Data"
    print "========================================"
    print "Corpus Precision {:.3}".format(prec)
    print "Corpus Recall    {:.3}".format(rec)
    print "Corpus F1        {:.3}".format(f1)
    print "----------------------------------------"
    print "TP: {} | FP: {} | FN: {}".format(TP, FP, FN)
    print "========================================\n"
    return map(lambda x: sorted(list(x)), [TP_set, FP_set, FN_set])

def get_implied_parts(part, doc, parts_by_doc):
    yield part
    if parts_by_doc:
        for p in parts_by_doc[doc]:
            if p.startswith(part) and len(part) >= 4:
                yield p

def parts_f1(candidates, gold_parts, parts_by_doc=None):
    parts = set()
    for c in candidates:
        doc = c.part.parent.document.name.upper()
        part = c.part.get_span()
        for p in get_implied_parts(part, doc, parts_by_doc):
            parts.add((doc, p))
    # parts = set([(c.part.parent.document.name.upper(), c.part.get_span()) for c in candidates])
    # import pdb; pdb.set_trace()
    TP_set = parts.intersection(gold_parts)
    TP = len(TP_set)
    FP_set = parts.difference(gold_parts)
    FP = len(FP_set)
    FN_set = gold_parts.difference(parts)
    FN = len(FN_set)
    prec = TP / float(TP + FP) if TP + FP > 0 else float('nan')
    rec  = TP / float(TP + FN) if TP + FN > 0 else float('nan')
    f1   = 2 * (prec * rec) / (prec + rec) if prec + rec > 0 else float('nan')
    print "========================================"
    print "Scoring on Entity-Level Gold Data"
    print "========================================"
    print "Corpus Precision {:.3}".format(prec)
    print "Corpus Recall    {:.3}".format(rec)
    print "Corpus F1        {:.3}".format(f1)
    print "----------------------------------------"
    print "TP: {} | FP: {} | FN: {}".format(TP, FP, FN)
    print "========================================\n"
    return map(lambda x: sorted(list(x)), [TP_set, FP_set, FN_set])


def candidate_to_entity(candidate):
    part = candidate.get_arguments()[0]
    attr = candidate.get_arguments()[1]
    doc  = part.parent.document.name
    return (doc.upper(), part.get_span().upper(), attr.get_span().upper())


def candidates_to_entities(candidates):
    entities = set()
    pb = ProgressBar(len(candidates))
    for i, c in enumerate(candidates):
        pb.bar(i)
        entities.add(candidate_to_entity(c))
    pb.close()
    return entities


def entity_to_candidates(entity, candidate_subset):
    matches = []
    for c in candidate_subset:
        c_entity = tuple([c[0].parent.document.name.upper()] + [c[i].get_span().upper() for i in range(len(c))])
        if c_entity == entity:
        # (part, attr) = c.get_arguments()
        # if (c[0].parent.document.name.upper(), part.get_span().upper(), attr.get_span().upper()) == entity:
            matches.append(c)
    return matches


def count_labels(entities, gold):
    T = 0
    F = 0
    for e in entities:
        if e in gold:
            T += 1
        else:
            F += 1
    return (T, F)


def part_error_analysis(c):
    print "Doc: %s" % c.part.parent.document
    print "------------"
    part = c.get_arguments()[0]
    print "Part:"
    print part
    print part.parent
    print "------------"
    attr = c.get_arguments()[1]
    print "Attr:"
    print attr
    print attr.parent
    print "------------"


def print_table_info(span):
    print "------------"
    if span.parent.table:
        print "Table: %s" % span.parent.table
    if span.parent.cell:
        print "Row: %s" % span.parent.row_start
        print "Col: %s" % span.parent.col_start
    print "Phrase: %s" % span.parent


def get_gold_parts_by_doc():
    gold_file = os.environ['SNORKELHOME'] + '/tutorials/tables/data/hardware/dev/hardware_dev_gold.csv'
    gold_parts = get_gold_dict(gold_file, doc_on=True, part_on=True, val_on=False)
    parts_by_doc = defaultdict(set)
    for part in gold_parts:
        parts_by_doc[part[0]].add(part[1])
    return parts_by_doc

def get_manual_parts_by_doc(documents):
    eeca_suffix = '^(A|B|C|R|O|Y|-?16|-?25|-?40)$'
    suffix_matcher = RegexMatchSpan(rgx=eeca_suffix, ignore_case=False)
    suffix_ngrams = OmniNgrams(n_max=1)
    part_ngrams = OmniNgramsPart(n_max=5)
    return generate_parts_by_doc(documents, 
                                 part_matcher=get_matcher('part'), 
                                 part_ngrams=part_ngrams, 
                                 suffix_matcher=suffix_matcher, 
                                 suffix_ngrams=suffix_ngrams)      


def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.'''
    # Code from http://stackoverflow.com/questions/38987/how-to-merge-two-python-dictionaries-in-a-single-expression
    # Note that the entries of Y will replace X's values if there is overlap.
    z = x.copy()
    z.update(y)
    return z

def generate_parts_by_doc(contexts, part_matcher, part_ngrams, suffix_matcher, suffix_ngrams):
    """
    Seeks to replace get_gold_dict by going through a first pass of the document
    and pull out valid part numbers.

    Note that some throttling is done here, but may be moved to either a Throttler
    class, or learned through using LFs. Throttling here just seeks to reduce
    the number of candidates produced.

    Note that part_ngrams should be at least 5-grams or else not all parts will
    be found.
    """
    suffixes_by_doc = defaultdict(set)
    parts_by_doc = defaultdict(set)

    print "Finding part numbers..."
    pb = ProgressBar(len(contexts))
    for i, context in enumerate(contexts):
        pb.bar(i)
        # extract parts
        for ts in part_ngrams.apply(context):
            # identify parts
            for pts in part_matcher.apply([ts]):
                parts_by_doc[pts.parent.document.name.upper()].add(pts.get_span())

            # identify suffixes
            for sts in suffix_matcher.apply([ts]):
                row_ngrams = set(get_row_ngrams(ts, infer=True))
                if ('classification' in row_ngrams or
                    'group' in row_ngrams or
                    'rank' in row_ngrams or
                    'grp.' in row_ngrams):
                    suffixes_by_doc[sts.parent.document.name.upper()].add(sts.get_span())
    pb.close()

    # Process suffixes and parts
    print "Appending suffixes..."
    final_dict = defaultdict(set)
    pb = ProgressBar(len(parts_by_doc))
    for doc in parts_by_doc.keys():
        pb.bar(i)
        for part in parts_by_doc[doc]:
            final_dict[doc].add(part)
            # TODO: This portion is really specific to our suffixes. Ideally
            # this kind of logic can be pased on the suffix_matcher that is
            # pass in. Or something like that...
            # The goal of this code is just to append suffixes to part numbers
            # that don't already have suffixes in a reasonable way.
            suffixes = suffixes_by_doc[doc]
            if not any(s in part[4:] for s in suffixes):
                for s in suffixes:
                    if s.isdigit(): s = '-' + s
                    final_dict[doc].add(part + s)
            # for suffix in suffixes:
            #     """
            #     if the part has no suffix, add it
            #     """
            #     if 
            #     if (suffix == "A" or suffix == "B" or suffix == "C"): # suffix.isalpha()?
            #         if not any(x in part[2:] for x in ['A', 'B', 'C']):
            #             final_dict[doc].add(part + suffix)
            #     else: # if it's 16/25/40
            #         if not suffix.startswith('-') and not any(x in part for x in ['16', '25', '40']):
            #             final_dict[doc].add(part + '-' + suffix)
            #         elif suffix.startswith('-') and not any(x in part for x in ['16', '25', '40']):
            #             final_dict[doc].add(part + suffix)
    pb.close()
    return final_dict