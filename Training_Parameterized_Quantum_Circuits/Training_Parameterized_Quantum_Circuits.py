## Necessario para plotar os circuitos e graficos bonitnho 
import matplotlib.pyplot as plt

from qiskit.circuit.library import RealAmplitudes
ansatz = RealAmplitudes(num_qubits=2, reps=1,
                        entanglement='linear').decompose()

ansatz.draw('mpl')
plt.show()


## Cria um observavel (Operador Hamiltoniano)
from qiskit.opflow import Z, I
hamiltonian = Z ^ Z
#print(type(hamiltonian))
print(hamiltonian.to_matrix())


## Monta a sanduiche do valor esperado 
## <ѱ|Ĥ|ѱ> 
from qiskit.opflow import StateFn, PauliExpectation
expectation = StateFn(hamiltonian, is_measurement=True) @ StateFn(ansatz)
pauli_basis = PauliExpectation().convert(expectation)

## Simula a medida do valor esperado
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.opflow import PauliExpectation, CircuitSampler
quantum_instance = QuantumInstance(Aer.get_backend('qasm_simulator'),
                                   # we'll set a seed for reproducibility
                                   shots = 8192, seed_simulator = 2718,
                                   seed_transpiler = 2718)
sampler = CircuitSampler(quantum_instance)

def evaluate_expectation(theta):
    value_dict = dict(zip(ansatz.parameters, theta))
    result = sampler.convert(pauli_basis, params=value_dict).eval()
    return np.real(result)

## Cria uma colecao randomica de tetas
import numpy as np
point = np.random.random(ansatz.num_parameters)
INDEX = 1
print()
print(point)

# calcula o gradiente como uma media 
EPS = 0.2
# make identity vector with a 1 at index ``INDEX``, otherwise 0
e_i = np.identity(point.size)[:, INDEX]
print(e_i)

plus = point + EPS * e_i
print(plus)
minus = point - EPS * e_i
print(minus)
finite_difference = (
    evaluate_expectation(plus) - evaluate_expectation(minus)) / (2 * EPS)
print(finite_difference)
