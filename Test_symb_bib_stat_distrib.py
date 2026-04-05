import sympy as sp

# Symbole definieren
J, T = sp.symbols('J T')

# Matrix P_DA
P_DA = sp.Matrix([
    [0, 0.5, 0.5, 0],
    [sp.exp(-J/T)*(1 - sp.Rational(1,2)*sp.exp(-J/T)), (1 - sp.exp(-J/T))**2, 0, sp.exp(-J/T)*(1 - sp.Rational(1,2)*sp.exp(-J/T))],
    [sp.exp(-J/T)*(1 - sp.Rational(1,2)*sp.exp(-J/T)), 0, (1 - sp.exp(-J/T))**2, sp.exp(-J/T)*(1 - sp.Rational(1,2)*sp.exp(-J/T))],
    [0, 0.5, 0.5, 0]
])

# Matrix P_DA
P_DA = sp.Matrix([
    [0, 0.5, 0.5, 0],
    [sp.exp(-J*2/T)*(1 - sp.Rational(1,2)*sp.exp(-J*2/T)), (1 - sp.exp(-J*2/T))**2, 0, sp.exp(-J*2/T)*(1 - sp.Rational(1,2)*sp.exp(-J*2/T))],
    [sp.exp(-J*2/T)*(1 - sp.Rational(1,2)*sp.exp(-J*2/T)), 0, (1 - sp.exp(-J*2/T))**2, sp.exp(-J*2/T)*(1 - sp.Rational(1,2)*sp.exp(-J*2/T))],
    [0, 0.5, 0.5, 0]
])
# Matrix P_DA
R_DA = sp.Matrix([
    [-sp.exp(+J/T), 0.5-sp.exp(J/T), 0.5-sp.exp(J/T), -sp.exp(J/T)],
    [sp.exp(-J*2/T)*(1 - sp.Rational(1,2)*sp.exp(-J*2/T))-sp.exp(-J/T), (1 - sp.exp(-J*2/T))**2-sp.exp(-J/T), -sp.exp(-J/T), sp.exp(-J*2/T)*(1 - sp.Rational(1,2)*sp.exp(-J*2/T))-sp.exp(-J/T)],
    [sp.exp(-J*2/T)*(1 - sp.Rational(1,2)*sp.exp(-J*2/T))-sp.exp(-J/T), -sp.exp(-J/T), (1 - sp.exp(-J*2/T))**2-sp.exp(-J/T), sp.exp(-J*2/T)*(1 - sp.Rational(1,2)*sp.exp(-J*2/T))-sp.exp(-J/T)],
    [-sp.exp(+J/T), 0.5-sp.exp(J/T), 0.5-sp.exp(J/T), -sp.exp(J/T)]
])

P, D = R_DA.diagonalize()
R_inv = P.inv()
print("P=")
for i in range(P.rows):
    print(P.row(i))
print("D=")
for i in range(D.rows):
    print(D.row(i))



# Vektor der stationären Verteilung-
v1, v2, v3, v4 = sp.symbols('v1 v2 v3 v4')
v = sp.Matrix([v1, v2, v3, v4])

# Gleichungen: (P - I)v = 0
eqs = list((P_DA.T - sp.eye(4))*v)
# Normalisierung: Summe der Einträge = 1
eqs.append(v1 + v2 + v3 + v4 - 1)

# Lösen symbolisch
solution_symbolic = sp.solve(eqs, [v1, v2, v3, v4], dict=True)

print("Stationäre Verteilung (symbolisch):")
print(solution_symbolic)
for i in solution_symbolic[0]:
    print(solution_symbolic[0][i])
if 0:
    if 0:
        # --- Numerische Auswertung ---
        # Beispielwerte für J und T
        J_val = 1.0
        T_val = 2.0

        # Ersetze J und T durch konkrete Werte
        solution_numeric = [{k: v.subs({J: J_val, T: T_val}).evalf() for k,v in sol.items()} 
                            for sol in solution_symbolic]

        print("\nStationäre Verteilung (numerisch) für J=1.0, T=2.0:")
        print(solution_numeric)