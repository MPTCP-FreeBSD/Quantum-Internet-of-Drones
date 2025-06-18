import math
import time
import pandas as pd
from itertools import combinations
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, state_fidelity, partial_trace
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error, amplitude_damping_error

nodes = [f'N{i}' for i in range(5)]
noise_model = 2
results = []

all_segments = []


for src, dst in combinations(nodes, 2):  # Total 10 links
    print("=== combinations: %d, %d",src,dst)
    segments = [f"{src}-R1", "R1-R2", f"R2-{dst}"]
    all_segments+=segments

# node_comm = (('N0','N1'),('N2','N3'))
# for src, dst in node_comm:  # Total 10 links
#     print("=== combinations: %d, %d",src,dst)
#     segments = [f"{src}-R1", "R1-R2", f"R2-{dst}"]
#     all_segments+=segments

print("all_segments: ",all_segments)

print("len(all_segments): ",len(all_segments))

unique_segments = list(set(all_segments))

print("unique_segments: ",unique_segments)

print("len(unique_segments): ",len(unique_segments))

segment_cross_talk_dict = {}

for segment in unique_segments:
    for othersegment in all_segments:
        if segment == othersegment:
            if segment_cross_talk_dict.get(segment) is None:
                segment_cross_talk_dict[segment] = 0
            else:
                segment_cross_talk_dict[segment]+=1


print()
print("segment_cross_talk_dict",segment_cross_talk_dict)