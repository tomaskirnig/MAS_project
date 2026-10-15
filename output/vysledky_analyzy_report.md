# Výsledky strukturální a komunitní analýzy sítě Facebook Large Page-Page Network
**Předmět:** MAS1 (Modelování a simulace / Analýza sítí)  
**Dataset:** Facebook Large Page-Page Webgraph (musae_facebook)  
**Datum generování:** 15.10.2026 08:27:10

---

## 1. Popis datového souboru
- **Zdroj:** Datová sada publikovaná v rámci studie *Multi-scale Attributed Node Embedding (Rozemberki et al., 2019)*.
- **Kontext:** Webgraph vzájemných sympatií („mutual likes“) mezi oficiálními ověřenými stránkami na Facebooku (listopad 2017).
- **Entita vrcholu:** Oficiální Facebook stránka (politici, vládní organizace, televizní pořady, firmy/společnosti).
- **Hrany:** Vzájemné lajky (neorientovaný graf, bez vah hran).
- **Atributy:** Kategorie stránky (`page_type`: government, company, politician, tvshow), textové příznaky z popisků.

---

## 2. Globální a strukturální vlastnosti sítě (Krok 2.a)

| Metrika | Hodnota | Interpretace |
| :--- | :--- | :--- |
| **Počet vrcholů (|V|)** | **22,470** | Celkový rozsah sítě ověřených stránek |
| **Počet hran (|E|)** | **170,823** | Celkový počet vzájemných propojení |
| **Hustota sítě (Density)** | **0.000677** | Typická řídká reálná sociální síť (density << 1) |
| **Průměrný stupeň** | **15.20** | Průměrně má stránka cca 15 sousedů |
| **Směrodatná odchylka stupně** | **26.41** | Vysoká variabilita, indikuje přítomnost velkých hubů |
| **Medián stupně** | **7.0** | Polovina stránek má 8 a méně propojení (asymetrie) |
| **Minimální stupeň** | **1** | Nejméně propojený uzel |
| **Maximální stupeň (Hub)** | **709** | Největší centrum propojení v síti |
| **Počet komponent** | **1** | Síť je plně souvislá (1 obří komponenta = 100 %) |
| **Průměrný shlukovací koef. (CC)** | **0.3597** | Vysoké lokální shlukování (typické pro sociální sítě) |
| **Globální tranzitivita** | **0.2323** | Podíl uzavřených trojúhelníků v celé síti |
| **Odhad průměrné délky cesty** | **4.93** | Vlastnost „malého světa“ (Small-World effect) |
| **Odhad průměru sítě (Diameter)** | **13** | Maximální zjištěný počet kroků mezi uzly |

### Interpretace distribuce stupňů a shlukovacího efektu:
1. **Distribuce stupňů (Degree Distribution):** Vykazuje těžký chvost (heavy-tail) s přibližným mocninným zákonem $P(k) \sim k^{-\gamma}$. Drtivá většina stránek má nízký stupeň, zatímco malé procento hubů má stovky až tisíce hran (bezškálová síť / Scale-Free).
2. **Shlukovací efekt C(k):** Průměrný shlukovací koeficient klesá s rostoucím stupněm uzlu $C(k) \sim k^{-alpha}$. To dokazuje **hierarchickou modularitu** sítě: malé uzly tvoří husté lokální klastry, zatímco velké huby propojují různé klastry navzájem a mají nižší lokální koeficient shlukování.

---

## 3. Analýza centralit a klíčové entity (Krok 2.a.iii)

### TOP 10 entit podle Degree Centrality (Nejvíce přímých propojení)
```
 Pořadí    ID                                   Název stránky  Kategorie  Stupeň    Skóre
      1 16895                                       U.S. Army government     709 0.031555
      2 19743                                 The White House government     678 0.030175
      3 21729                           The Obama White House government     659 0.029329
      4 14497                        U.S. Army Chaplain Corps government     650 0.028929
      5  1387 Honolulu District, U.S. Army Corps of Engineers government     504 0.022431
      6 10379                        U.S. Department of State government     468 0.020829
      7 19347        FEMA Federal Emergency Management Agency government     448 0.019939
      8 21120                             European Parliament government     417 0.018559
      9  2442                     Army Training Network (ATN) government     408 0.018158
     10  8139                       Defense Commissary Agency government     387 0.017224
```

### TOP 10 entit podle Betweenness Centrality (Klíčové mosty a informační toky)
```
 Pořadí    ID            Název stránky  Kategorie  Stupeň    Skóre
      1   701                 Facebook    company     380 0.111830
      2 11003             Barack Obama politician     341 0.087209
      3 19743          The White House government     678 0.038109
      4 21729    The Obama White House government     659 0.037728
      5 21120      European Parliament government     417 0.033251
      6 18819              Niels Annen politician     199 0.026812
      7 13294      European Commission government     330 0.019748
      8 17983                      CNN    company     167 0.018722
      9  8482                     NATO government     222 0.017390
     10 10379 U.S. Department of State government     468 0.016649
```

### TOP 10 entit podle Closeness Centrality (Nejrychlejší dosah k ostatním uzlům)
```
 Pořadí    ID              Název stránky  Kategorie  Stupeň    Skóre
      1 19743            The White House government     678 0.322581
      2 21729      The Obama White House government     659 0.322581
      3   701                   Facebook    company     380 0.322581
      4 11003               Barack Obama politician     341 0.319149
      5 22171        U.S. Embassy Ottawa government     330 0.307377
      6 10379   U.S. Department of State government     468 0.306748
      7 17983                        CNN    company     167 0.305499
      8  8482                       NATO government     222 0.301205
      9 16895                  U.S. Army government     709 0.298805
     10 11657 U.S. Embassy in Mozambique government     199 0.298211
```

### TOP 10 entit podle PageRanku (Globální autority sítě)
```
 Pořadí    ID          Název stránky  Kategorie  Stupeň    Skóre
      1   701               Facebook    company     380 0.001325
      2 16809 Sir Peter Bottomley MP politician     174 0.001208
      3 10440           Harish Rawat politician      44 0.001057
      4 19743        The White House government     678 0.001011
      5 21729  The Obama White House government     659 0.000984
      6 11003           Barack Obama politician     341 0.000922
      7 20415       Joachim Herrmann politician     320 0.000921
      8 16895              U.S. Army government     709 0.000896
      9 21120    European Parliament government     417 0.000878
     10 14650          Manfred Weber politician     326 0.000865
```

### Interpretace centralit:
- **Facebook (ID 701)** a klíčové státní instituce (např. **Barack Obama ID 11003**, **The White House ID 19743**, **U.S. Army ID 16895**, **European Parliament ID 21120**) drží naprosté prvenství v Betweenness a PageRanku.
- Tvoří centrální informační dálnice a spojnice mezi různými segmenty sítě (politika, instituce, média).

---

## 4. Analýza komunitní struktury (Krok 2.b)

Porovnání dvou odlišných komunitních algoritmů:
1. **Louvain:** Heuristická optimalizace modularity na více úrovních.
2. **Label Propagation (LPA):** Asynchronní šíření lokálních jmenovek uzly.

| Metrika | Algoritmus 1: Louvain | Algoritmus 2: Label Propagation (LPA) |
| :--- | :--- | :--- |
| **Hodnota modularity (Q)** | **0.8074** | **0.7340** |
| **Počet nalezených komunit** | **60** | **1411** |
| **Maximální velikost komunity** | **3,830** uzlů | **3,080** uzlů |
| **Minimální velikost komunity** | **6** uzlů | **2** uzlů |
| **Průměrná velikost komunity** | **374.5** uzlů | **15.9** uzlů |
| **Medián velikosti komunity** | **154.0** uzlů | **4.0** uzlů |

### Interpretace komunitní struktury:
- **Louvain algoritmus** dosahuje velmi vysoké modularity ($Q = 0.8074 > 0.8$), což značí silnou a přirozeně separovanou komunitní strukturu. Vytvořil přibližně 60 makro-komunit, které reflektují tematické a institucionální zaměření stránek.
- **Label Propagation (LPA)** dosahuje modularity $Q = 0.7340$ a detekuje velké množství mikro-komunit (1411), což je typické pro lokální algoritmy bez globální optimalizace modularity.

---

## 5. Přehled vygenerovaných vizualizací
Všechny grafy jsou uloženy v adresáři `output/`:
1. `1_distribuce_stupnu.png` - Histogram stupňů + Log-log distribuce (Power-Law).
2. `3_shlukovaci_efekt.png` - Průměrný shlukovací koeficient C(k) v závislosti na stupni k (Hierarchická modularita).
3. `4_distribuce_centralit.png` - Distribuce jednotlivých měr centralit.
4. `5_vizualizace_center_hubu.png` - Podgraf klíčových hubů s popiskami a barvami podle kategorie stránek.
5. `6_distribuce_komunit.png` - Distribuce a porovnání velikostí komunit (Louvain vs LPA).
6. `7_vizualizace_komunit.png` - Vizualizace jádra sítě obarvená podle komunit.
7. `facebook_network_full.gexf` - Plný export do Gephi se všemi atributy a metrikami.
