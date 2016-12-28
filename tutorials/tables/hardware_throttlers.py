from snorkel.lf_helpers import *
from collections import namedtuple
from hardware_lfs import *

"""
NOTE: Throttlers must operate on multiple component Spans of a candidate. If the body
of a throttler operates only on a single Span, move that logic to a matcher instead
(likely a LambdaFunctionMatch).
"""

throttlers = {}

# Use this thin wrapper when using LFs that need 'part' and 'attr' attributes
FakeCandidate = namedtuple('FakeCandidate', ['part', 'attr'])
# c = FakeCandidate(part, attr)

def same_page_throttler((part, attr)):
#    if not same_page((part, attr)): return False
    if same_table((part, attr)):
        return (is_horz_aligned((part, attr)) or is_vert_aligned((part, attr)))
    if LF_part_miss_match((part, attr)) < 0: return False
    return True
# aligned or part is global

def polarity_throttler((part, attr)):
#    if not same_page((part, attr)): return False
#    if LF_part_miss_match_part((part, attr)) < 0: return False
    if LF_part_miss_match_polarity((part, attr)) < 0: return False
    if same_table((part, attr)):
        return (is_horz_aligned((part, attr)) or is_vert_aligned((part, attr)))
    return True

throttlers['ce_v_max'] = same_page_throttler
throttlers['polarity'] = polarity_throttler

def get_throttler(attr):
    # return None
    if attr in ['ce_v_max', 'polarity']:
        return throttlers[attr]
    else:
        return None

# def get_part_throttler_wrapper():
#     """get a part throttler wrapper to throttler unary candidates with the usual binary throttler"""
#     def part_throttler_wrapper(part):
#         return part_throttler((part[0], None))
#     return part_throttler_wrapper

# def get_part_throttler():
#     return part_throttler

# def part_throttler((part, attr)):
#     """throttle parts that are in tables of device/replacement parts"""
#     aligned_ngrams = set(get_aligned_ngrams(part))
#     if (overlap(['replacement'], aligned_ngrams) or
#         (len(aligned_ngrams) > 25 and 'device' in aligned_ngrams) or
#         # CentralSemiconductorCorp_2N4013.pdf:
#         get_prev_sibling_tags(part).count('p') > 25 or
#         overlap(['complementary', 'complement', 'empfohlene'], 
#                 chain.from_iterable([
#                     get_left_ngrams(part, window=10),
#                     get_aligned_ngrams(part)]))):
#         return False
#     else:
#         return True
