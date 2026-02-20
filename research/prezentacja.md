---
marp: true
theme: gaia
_class: lead
paginate: true
backgroundColor: #fff
backgroundImage: url('https://marp.app/assets/hero-background.svg')
style: |
  section { font-family: 'Arial', sans-serif; padding:30px }
  h1 { color: #2c3e50; }
  h2 { color: #e74c3c; }
  strong { color: #2980b9; }
---

# Podstawy kompresji LLM + kodowanie arytmetyczne

Wstęp: dwa kluczowe elementy

- Model językowy
- Koder arytmetyczny

---

# Model językowy

- Rozumie język i kontekst.
- Podaje prawdopodobieństwa następnej litery/tokenu — jeśli było „Ala ma”, to P("kota") = 0.9, a inne tokeny mają małe PPB.


---

# Koder arytmetyczny

- Upycha bity jak najoszczędniej.
- Korzysta z PPB od modelu językowego.
- Im większe PPB danej litery/tokenu, tym mniej bitów zajmuje.

---

# Czym jest kodowanie arytmetyczne?

To metoda zamiany ciągu symboli na jedną liczbę z przedziału $[0,1)$.

Każdy kolejny symbol zawęża przedział.

---

# Przykład: kodujemy „BA”

Alfabet: {A, B, C}

Model podaje:

- $P(A)=0.5$
- $P(B)=0.25$
- $P(C)=0.25$

Startujemy od przedziału $[0,1)$.

---

# Krok 1: litera „B”

Dzielimy $[0,1)$ wg PPB:

- A: $[0.0, 0.5)$
- B: $[0.5, 0.75)$
- C: $[0.75, 1.0)$

Pierwszy symbol to B, więc nowy przedział to $[0.5, 0.75)$.

---

# Krok 2: litera „A”

Bierzemy przedział $[0.5, 0.75)$ i znów dzielimy (dla uproszczenia: te same PPB):

- A wewnątrz tego zakresu: $[0.5, 0.625)$

Dla słowa „BA” końcowy zakres to $[0.5, 0.625)$.

Aby zapisać „BA”, wybieramy dowolną liczbę z tego przedziału.


---
# Czemu to oszczędza?
Zauważmy, że jeśli mamy bardzo wąski przedział, np. 0.0625–0.0650, potrzebujemy dużo bitów, aby go zapisać — np. 0.0001–0.10(1001).
Jeśli mamy przedział 0.0–0.5, to wystarczy jeden bit (0–0.1).