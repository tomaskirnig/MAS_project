# 🧠 Velký tahák k projektu MAS1: Analýza sítě Facebooku
> **Cíl taháku:** Vysvětlit každou metriku, vzorec a graf tak jednoduše a polopatě, aby to pochopil naprosto každý a byl schopný to s přehledem obhájit u zkoušky nebo prezentace.

---

## 0. O čem ten dataset vůbec je? (Sedlácký rozum)
* **Co zkoumáme:** Síť ověřených stránek na Facebooku z roku 2017 (`musae_facebook`).
* **Vrchol (Uzel / Node):** Jedna oficiální facebooková stránka (např. *Barack Obama*, *Facebook*, *The White House*, *CNN*, *European Parliament*). Celkem **22 470 stránek**.
* **Hrana (Spojení / Edge):** Vzájemný lajk („Mutual Like“) mezi dvěma stránkami. Když si stránka A i stránka B daly lajk, existuje mezi nimi čára. Celkem **170 823 hran**.
* **Je síť orientovaná?** Ne, hrany nemají šipky (je to neorientovaný graf).
* **Kategorie stránek:** Každá stránka patří do jedné ze 4 škatulek:
  1. `politician` (politici)
  2. `government` (státní a mezinárodní instituce, armáda, ambasády)
  3. `company` (firmy, značky)
  4. `tvshow` (televizní pořady a média)

---

## 1. Globální vlastnosti sítě

### A. Počet vrcholů ($|V|$) a hran ($|E|$)
* **Naše hodnoty:** $|V| = 22\,470$, $|E| = 170\,823$.
* **Polopatě:** Velikost hřiště. Kolik máme hráčů a kolik mezi nimi vede šňůr.

### B. Hustota sítě (Density $\rho$)
* **Vzorec:** $\rho = \frac{2 \cdot |E|}{|V| \cdot (|V| - 1)}$
* **Naše hodnota:** $0.000677$ (cca 0.07 %).
* **Polopatě:** Kolik procent ze všech teoreticky možných propojení reálně existuje.
  * Kdyby každá stránka lajkovala úplně každou, byla by hustota $1.0$ (100 %).
  * Naše síť má hustotu 0.07 %. To je typická **řídká síť (sparse network)**. V reálném světě nikdo nezná každého.

### C. Stupeň vrcholu (Degree $k$)
* **Co to je:** Počet přímých sousedů (kolik lajků daná stránka má).
* **Naše hodnoty:**
  * **Průměrný stupeň $\langle k \rangle$:** $15.20$ (průměrná stránka má cca 15 sousedů).
  * **Medián stupně:** $7.0$.
  * **Minimální stupeň:** $1$ (stránka s jedním spojením).
  * **Maximální stupeň (Hub):** $709$ (U.S. Army).
* **Proč je průměr (15) mnohem větší než medián (7)?**
  * Přesně polovina stránek má 7 a méně sousedů!
  * Průměr táhne nahoru pár gigantických celebrit (tzv. **hubů**), které mají stovky propojení. Je to jako s průměrným platem, když v místnosti sedí Elon Musk.

### D. Distribuce stupňů a bezškálovost (Scale-Free & Power-Law)
* **Graf:** `1_distribuce_stupnu.png`.
* **Polopatě:**
  * V normálním světě (např. výška lidí) máme Gaussovu křivku (většina lidí měří kolem 175 cm, nikdo nemá 5 metrů).
  * V komplexních sítích platí **pravidlo těžkého chvostu (Heavy-Tail)**: Drtivá většina uzlů má jen pár přátel, ale existuje pár monster (hubů), co jich mají tisíce.
  * Když vyneseme pravděpodobnost $P(k)$ vůči stupni $k$ v **logaritmickém měřítku (Log-Log)**, vyjde klesající **přímka**. To značí mocninný zákon: $P(k) \sim k^{-\gamma}$.
  * Síť je tedy **bezškálová (Scale-Free)**.

---

## 2. Souvislost a vzdálenosti v síti

### A. Souvislé komponenty (Connected Components)
* **Naše hodnota:** $1$ komponenta (100 % sítě).
* **Polopatě:** Síť se nerozpadá na izolované ostrovy. Z libovolné stránky se po hranách dá doklikat na jakoukoliv jinou stránku v síti.

### B. Průměrná délka cesty (Average Path Length)
* **Naše hodnota:** cca $4.93$ kroků.
* **Polopatě:** Fenomén **malého světa (Small-World effect)** neboli *„Šest stupňů odloučení“*.
  * Ačkoliv je v síti přes 22 tisíc stránek, jakékoliv dvě náhodné stránky dělí v průměru necelých **5 kliknutí**.

### C. Průměr sítě (Diameter)
* **Naše hodnota:** $13$.
* **Polopatě:** Nejdelší ze všech nejkratších cest. Dva nejvzdálenější outsidery na opačných koncích světa dělí maximálně 13 kliknutí.

---

## 3. Shlukování (Clustering)

### A. Průměrný shlukovací koeficient (Clustering Coefficient $\langle CC \rangle$)
* **Vzorec pro uzel $i$:** $C_i = \frac{2 \cdot e_i}{k_i \cdot (k_i - 1)}$  
  *(kde $e_i$ je počet propojení mezi sousedy uzlu $i$, a jmenovatel je max možný počet propojení mezi nimi).*
* **Naše hodnota:** $\langle CC \rangle = 0.3597$ (cca 36 %).
* **Polopatě:** Otázka: *„Kamarádí se moji kamarádi mezi sebou?“*
  * V náhodném grafu by byl shlukovací koeficient roven hustotě sítě ($0.0006$).
  * U nás je $0.36$, což je **600× více než náhoda!** Lidé a instituce tvoří uzavřené party (trojúhelníky).

### B. Globální tranzitivita
* **Naše hodnota:** $0.2323$ ($23.2\ \%$).
* **Polopatě:** Podíl uzavřených trojúhelníků v celé síti vůči všem otevřeným trojicím.

### C. Shlukovací efekt $C(k)$ vůči stupni $k$ (Hierarchická struktura)
* **Graf:** `3_shlukovaci_efekt.png`.
* **Polopatě:**
  * **Malé uzly ($k < 10$):** Mají obrovský shlukovací koeficient ($C \approx 0.6 - 0.8$). Žijí ve své malé uzavřené komunitě, kde se všichni navzájem znají.
  * **Velké huby ($k > 100$):** Mají malý shlukovací koeficient ($C \approx 0.05$).
  * **Proč to tak je?** Protože hub (třeba *Facebook* nebo *Obama*) propojuje úplně cizí skupiny (např. rybáře v Ohiu a politiky v Bruselu). Tito jeho sousedé se mezi sebou neznají a nekamarádí.
  * Pokles $C(k) \sim k^{-1}$ je učebnicovým důkazem **hierarchické modularity sítě**.

---

## 4. Centrality (Kdo je v síti kápo?)

Centralita je číslo, které říká, jak moc je daný uzel důležitý. Existují různé druhy důležitosti:

| Typ centrality | Polopatické vysvětlení | Co to znamená v praxi | Vítěz v našich datech |
| :--- | :--- | :--- | :--- |
| **Degree Centrality** | **Kdo má nejvíc přátel.** Pouhý počet hran napojených na uzel. | Nejoblíbenější stránka, má největší okamžitý dosah. | **U.S. Army** (709 spojení)<br>**The White House** (678) |
| **Betweenness Centrality** (Mezilehlost) | **Kdo je hlídač mostu.** Počítá, kolik nejkratších cest mezi všemi dvojicemi uzlů musí projít přes tento uzel. | Klíčový zprostředkovatel toku informací. Kdyby tento uzel zmizel, síť se rozpadne na nesouvisející kusy. | **Facebook** (skóre 0.1118)<br>**Barack Obama** (skóre 0.0872) |
| **Closeness Centrality** (Blízkost) | **Kdo má ke všem nejblíže.** Převrácená hodnota průměrné vzdálenosti k ostatním uzlům v síti. | Kdo dokáže nejrychleji rozšířit drb nebo informaci do celé sítě bez prostředníků. | **The White House** (0.323)<br>**Facebook** (0.323) |
| **PageRank / Eigenvector** | **Kdo zná vlivné lidi.** Nestačí mít hodně přátel; záleží na tom, zda tví přátelé jsou sami důležití. | Globální autorita. Odkaz od Obamy má větší váhu než odkaz od 100 neznámých lokálních stránek. | **Facebook** (0.0013)<br>**Sir Peter Bottomley** (0.0012) |

---

## 5. Komunitní detekce (Kmeny a bubliny)

* **Co je komunita:** Skupina uzlů, které mají mezi sebou obrovské množství propojení, ale směrem ven do zbytku sítě jich mají málo.
* **Modularita ($Q$):** Metrika kvality rozdělení do komunit.
  * Pohybuje se od $-0.5$ do $+1.0$.
  * Hodnoty nad $0.3$ už značí reálné komunity.
  * Hodnoty **nad 0.7 jsou extrémně vysoké** — síť je jasně rozpadlá do pevných názorových / tematických bloků.

### Porovnání dvou algoritmů v našem projektu:

| Vlastnost | 1. Louvain algoritmus | 2. Label Propagation (LPA) |
| :--- | :--- | :--- |
| **Jak funguje?** | Globální optimalizace modularity. Postupně spojuje uzly do shluků tak, aby maximalizoval celkové číslo $Q$. | Lokální šíření jmenovek. Každý uzel se podívá na sousedy a přebarví se na barvu, kterou má většina z nich (jako volby). |
| **Dosažená modularita $Q$** | **$0.8074$** (skvělé, velmi čisté komunity) | **$0.7340$** (velmi dobré) |
| **Počet komunit** | **60 velkých komunit** | **1 411 menších mikro-komunit** |
| **Největší komunita** | 3 830 stránek | 3 080 stránek |
| **Průměrná velikost** | 374 uzlů | 16 uzlů |
| **Zhodnocení** | Vhodný pro nalezení hlavních velkých segmentů (např. americká politika, evropské instituce, armáda, televize). | Má tendenci rozpadat se na stovky drobných střípků a rodinek na periferii. |

---

## 6. Co znamenají vizualizace v `output/`?

1. **`1_distribuce_stupnu.png`**  
   Vlevo normální histogram (vidíš obrovský kopec u nuly a pak dlouhou prázdnotu doprava). Vpravo log-log graf (krásná rovná šikmá linie = potvrzení Scale-Free vlastnosti).
2. **`3_shlukovaci_efekt.png`**  
   Graf ukazuje, jak shlukovací koeficient padá dolů se stoupajícím stupněm $k$. Důkaz: malé stránky jsou izolované kroužky, velké stránky jsou otevřené mosty.
3. **`4_distribuce_centralit.png`**  
   Ukazuje, že vysokou betweenness nebo pagerank má jen naprosté promile uzlů. Zbytek sítě jsou pasivní konzumenti.
4. **`5_vizualizace_center_hubu.png`**  
   Výřez TOP 70 nejsilnějších hubů. Barvy reprezentují typ: modrá = instituce/vláda, zelená = firmy, červená = politici, oranžová = TV pořady.
5. **`6_distribuce_komunit.png`**  
   Křivka velikosti komunit. U Louvainu vidíme cca 15 gigantických komunit a pak menší. U LPA vidíme dlouhý pád stovek mikroskopických skupinek.
6. **`7_vizualizace_komunit.png`**  
   Jádro sítě, kde barva uzlu neurčuje typ stránky, ale její příslušnost k Louvain komunitě. Vidíš, jak se uzly stejné barvy přirozeně shlukují k sobě.
7. **`facebook_network_full.gexf`**  
   Datový soubor pro program **Gephi**. Když ho tam otevřeš, můžeš si síť interaktivně prohlížet, barvit a zoomovat.

---

## 7. Rychlý návod: Jak to obhájit za 2 minuty u tabule
1. *„Analyzovali jsme síť vzájemných sympatií mezi 22 470 ověřenými Facebook stránkami.“*
2. *„Síť je typický bezškálový malý svět (Scale-Free Small-World): je velmi řídká (cca 0.07 %), ale k jakékoliv stránce se dostanete v průměru na 5 kliknutí.“*
3. *„Stupně uzlů se řídí mocninným zákonem: většina stránek má pár kontaktů, zatímco huby jako U.S. Army nebo Bílý dům jich mají přes 600.“*
4. *„V centralitě mezilehlosti (Betweenness) zcela dominuje stránka Facebooku a Barack Obama – fungují jako hlavní informační brány mezi odlišnými světy.“*
5. *„Shlukovací efekt dokázal hierarchickou modularitu: lokální partičky mají vysoké CC, ale centrální huby spojují nesouvisející komunity.“*
6. *„Detekce komunit pomocí algoritmu Louvain dosáhla vysoké modularity $Q = 0.81$, což potvrzuje, že stránky na sociální síti tvoří silně uzavřené tématické a institucionální bubliny.“*
