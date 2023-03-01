# Filename: cider.py
#
# Description: Describes the class to compute the CIDEr (Consensus-Based Image Description Evaluation) Metric 
#               by Vedantam, Zitnick, and Parikh (http://arxiv.org/abs/1411.5726)
#
# Creation Date: Sun Feb  8 14:16:54 2015
#
# Authors: Ramakrishna Vedantam <vrama91@vt.edu> and Tsung-Yi Lin <tl483@cornell.edu>

from .cider_scorer import CiderScorer
import pdb

class Cider:
    """
    Main Class to compute the CIDEr metric 

    """
    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        # set cider to sum over 1 to 4-grams
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma

    def compute_score(self, gts, res):
        """
        Main function to compute CIDEr score
        :param  hypo_for_image (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                ref_for_image (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: cider (float) : computed CIDEr score for the corpus 
        """

        assert(sorted(gts.keys()) == sorted(res.keys()))
        imgIds = sorted(gts.keys())

        cider_scorer = CiderScorer(n=self._n, sigma=self._sigma)

        for id in imgIds:
            hypo = res[id]
            ref = gts[id]
            #print(hypo)
            #print(ref)
            #['a little girl is playing with her toys']
            #['a baby with pink dress playing in front of the camera', 'a small girl holding milk bottle and comb doing some actions', 'a kid with feeding bottle is playing on a room', 'a baby in rose walking and smiling with her mother', 'a baby dressed in a pink outfit is carrying a bottle of milk and a blue brush', 'the little girl dress in the pink out fit hold the bottle and hit with the brush', 'a girl toddler in a pink outfit walking around with a bottle', 'a lady is speaking about a baby girl the baby girl hits someone with a brush and the lady says that is not ok', 'the cute little girl in pink is being asked by her mother to show her teeth', 'a baby in pink dress holding a feeding bottle and come to beat his parent', 'a barefoot toddler wearing a pink romper with gathered fabric hits a woman with a blue hairbrush and tries to amend with a kiss', 'a mother ask her daughter in her play room to see her daughter teeth', 'a woman and a baby in a home talking', 'a lady is asking the baby girl with a pick outfit on name alyah to let her see her teeth', 'a little girl in pink is walking through a room', 'a baby in pink dress walking on floor brush in hand person beside walking displaying on screen', 'a girl toddler wearing pink overalls is holding a bottle in her arms while swinging a brush', 'a cute little girl is wearing a nice pink dress', 'baby in pink tshirt is playing with the camera', 'young child walking around with a jar and a brush']

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) >= 1)

            cider_scorer += (hypo[0], ref)

        (score, scores) = cider_scorer.compute_score()

        return score, scores

    def method(self):
        return "CIDEr"
