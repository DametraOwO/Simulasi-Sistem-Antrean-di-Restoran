from numpy import array, dot, isclose
from matplotlib.pyplot import plot, show, figure, title, xlabel, ylabel, legend, grid
from IPython.display import Math, display
import sympy as sym
from numpy.linalg import eig

# Matriks Transisi untuk Sistem Antrean di Restoran
P = array([[0.6, 0.3, 0.1],
           [0.2, 0.5, 0.3],
           [0.1, 0.4, 0.5]])
print("Matriks P =")
display(Math(sym.latex(sym.simplify(P))))

# Kondisi awal (Kosong)
P0 = array([1, 0, 0])
print("Matriks P0 =")
display(Math(sym.latex(sym.simplify(P0))))

# Prediksi untuk 10 hari ke depan
P1 = dot(P0, P)
print("Matriks P1 =")
display(Math(sym.latex(sym.simplify(P1))))

P_2 = dot(P, P)
print("Matriks P_2 =")
P2 = dot(P0, P_2)
display(Math(sym.latex(sym.simplify(P2))))

P_3 = dot(P_2, P)
print("Matriks P_3 =")
P3 = dot(P0, P_3)
display(Math(sym.latex(sym.simplify(P3))))

P_4 = dot(P_3, P)
print("Matriks P_4 =")
P4 = dot(P0, P_4)
display(Math(sym.latex(sym.simplify(P4))))

P_5 = dot(P_4, P)
print("Matriks P_5 =")
P5 = dot(P0, P_5)
display(Math(sym.latex(sym.simplify(P5))))

P_6 = dot(P_5, P)
print("Matriks P_6 =")
P6 = dot(P0, P_6)
display(Math(sym.latex(sym.simplify(P6))))

P_7 = dot(P_6, P)
print("Matriks P_7 =")
P7 = dot(P0, P_7)
display(Math(sym.latex(sym.simplify(P7))))

P_8 = dot(P_7, P)
print("Matriks P_8 =")
P8 = dot(P0, P_8)
display(Math(sym.latex(sym.simplify(P8))))

P_9 = dot(P_8, P)
print("Matriks P_9 =")
P9 = dot(P0, P_9)
display(Math(sym.latex(sym.simplify(P9))))

# Menampilkan hasil prediksi dalam grafik
figure(figsize=(10, 6))
days = range(1, 11)  # 10 hari
kosong = [P0[0], P1[0], P2[0], P3[0], P4[0], P5[0], P6[0], P7[0], P8[0], P9[0]]
sedikit_antrean = [P0[1], P1[1], P2[1], P3[1], P4[1], P5[1], P6[1], P7[1], P8[1], P9[1]]
banyak_antrean = [P0[2], P1[2], P2[2], P3[2], P4[2], P5[2], P6[2], P7[2], P8[2], P9[2]]

plot(days, kosong, label='Kosong', marker='o')
plot(days, sedikit_antrean, label='Sedikit Antrean', marker='o')
plot(days, banyak_antrean, label='Banyak Antrean', marker='o')

title('Simulasi Sistem Antrean di Restoran Menggunakan Rantai Markov (10 Hari)')
xlabel('Hari')
ylabel('Probabilitas')
legend()
grid(True)
show()

# Mencari Titik Equilibrium
eigenvalues, eigenvectors = eig(P.T)
steady_state = eigenvectors[:, isclose(eigenvalues, 1)]

# Normalisasi agar jumlah probabilitasnya 1
steady_state = steady_state / steady_state.sum(axis=0)
print("Steady State Vector =")
display(Math(sym.latex(sym.simplify(steady_state))))

# Verifikasi Equilibrium
equilibrium = dot(steady_state.T, P)
print("Equilibrium Verification =")
display(Math(sym.latex(sym.simplify(equilibrium))))
