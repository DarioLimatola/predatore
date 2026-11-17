"""Standalone XOR example: no simulator, GUI or saved profile."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from boolean_engine import BinaryVectorNetwork

PATTERNS=((0,0),(0,1),(1,0),(1,1))


def accuracy(net):
    clone=BinaryVectorNetwork.restore(json.loads(json.dumps(net.export())))
    clone.plasticity_enabled=False
    hits=0
    for a,b in PATTERNS:
        clone.reset()
        hits+=clone.decide((a,b))==(a^b)
    return hits/len(PATTERNS)


def main():
    net=BinaryVectorNetwork(seed=19,sensors=2,action_bits=1,neurons=4,shape=(8,),
                            memory_capacity=32,context_history=1)
    before=accuracy(net)
    for _ in range(12):
        for a,b in PATTERNS:
            net.reset()
            action=net.decide((a,b))
            net.transition((a,b))
            net.feedback(1 if action==(a^b) else -1)
    print(json.dumps({'initial_accuracy':before,'final_accuracy':accuracy(net),'contexts':len(net.contextual_memory.rows)}))


if __name__=='__main__':main()
