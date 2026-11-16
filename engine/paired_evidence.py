"""Paired Boolean outcome diagnostics, independent of the candidate policy.

The exact sign-test tail is an integer numerator over a power-of-two denominator.
It is diagnostic evidence on fixed paired seeds, not a non-inferiority guarantee
and not a correction for repeatedly selecting candidates on the same validation set.
"""
from math import comb


def paired_capture_evidence(reference, candidate):
    a=reference.get('paired_outcomes');b=candidate.get('paired_outcomes')
    if not a or not b or a['episodes']!=b['episodes']:
        return None
    n=a['episodes'];mask=(1 << n)-1
    previous=a['captures'] & mask;current=b['captures'] & mask
    gains=current & ~previous & mask
    losses=previous & ~current & mask
    wins=gains.bit_count();lost=losses.bit_count();discordant=wins+lost
    numerator=sum(comb(discordant,k) for k in range(wins,discordant+1))
    denominator=1 << discordant
    return {'episodes':n,'gain_mask':gains,'loss_mask':losses,
            'gained_captures':wins,'lost_captures':lost,
            'capture_delta_numerator':wins-lost,'capture_delta_denominator':n,
            'discordant_pairs':discordant,
            'one_sided_sign_p_numerator':numerator,
            'one_sided_sign_p_denominator':denominator,
            'note':'Exact paired sign tail under exchangeable independent discordant pairs; diagnostic only.'}
