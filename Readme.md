# QM + Dual-Complex Numbers

## Read the paper (WIP)

```sh
git clone https://github.com/Bagoult/dual-quantum.git
make
open Paper/main.pdf
```

## Play with the Sage file

```sh
mamba activate sage
sage
```

Once inside of sage, type

```py
load("dual_comp.sage")
```

Then you can play around

```py
Psi = np.array([1 + 2 * ϵ, 1 - ϵ, 1 - ϵ])
ψ = Psi / norm(Psi)
one, two, three = (np.eye(1, 3, i) for i in range(3))
A = outer(one, one)
B = outer(two, two) + outer(three, three)
M1 = A + ϵ * B
M2 = B + ϵ * A

# M1, M2 is a valid measurement operators collection
assert np.all(completeness(M1, M2) - np.identity(3) == 0)

p1 = measure_prob(M1, ψ)
p2 = measure_prob(M2, ψ)
ψ_1 = M1 @ ψ
ψ_2 = M2 @ ψ
```
