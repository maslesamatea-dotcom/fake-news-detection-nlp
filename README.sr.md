# Detekcija lažnih vijesti (NLP + mašinsko učenje)

**Jezik:** [English](README.md) | Srpski

Projektni rad iz predmeta Veštačka inteligencija
- **Autor:** Matea Masleša
- **Predmetni nastavnik:** dr Đorđe Petrović
- **Škola:** Visoka škola strukovnih studija za informacione tehnologije (ITS), Beograd, jun 2026.

## Cilj

Binarna klasifikacija teksta: da li je vijest **lažna (fake)** ili **istinita (real)**. Poređeni su TF-IDF vektorizacija i tri klasifikatora: Logistička regresija, Naive Bayes i Random Forest.

## Podaci

- **Izvor:** Kaggle, *Fake and Real News Dataset* ([link](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)). Podaci nisu u repozitoriju (vidi `data/README.md`).
- **Veličina:** 23.481 lažnih i 21.417 istinitih vijesti (ukupno 44.898), na engleskom jeziku, uglavnom političke teme iz perioda 2015–2018, s fokusom na američke izbore 2016.
- **Kolone:** naslov, tekst, kategorija i datum. Kategorija i datum se ne koriste kao feature-i, model uči samo iz teksta.
- **Čišćenje skupa:** uklonjeno 6.252 duplikata po tekstu (preostalo 38.646), pa zapisi sa tekstom kraćim od 100 i naslovom kraćim od 5 karaktera. Konačni skup ima **38.343** vijesti: 30.674 za trening (13.721 lažnih, 16.953 istinitih) i 7.669 za test (3.431 lažnih, 4.238 istinitih).

## Metodologija

1. **Priprema teksta** (`clean_text`): mala slova, uklanjanje URL-ova, uklanjanje oznake agencije *Reuters* (da model ne uči izvor umjesto sadržaja), uklanjanje znakova i brojeva, stopwords (NLTK), tokena kraćih od 3 slova i lematizacija (WordNet).
2. **Podjela:** 80/20, stratifikovana, `random_state=42`. Vektorizator se uči (`fit`) samo na trening skupu, a na testu se primjenjuje samo `transform`, da se izbjegne curenje podataka.
3. **TF-IDF:** `max_features=30000`, `ngram_range=(1, 2)`, `min_df=3`, `sublinear_tf=True`.
4. **Modeli:** Logistička regresija (`C=1.0`, L2), Multinomijalni Naive Bayes (`alpha=0.1`), Random Forest (200 stabala, `max_depth=30`, `min_samples_leaf=2`).
5. **Evaluacija:** hold-out na test skupu (accuracy, precision, recall, F1, ROC-AUC, matrica konfuzije) i kvalitativna analiza grešaka.

## Rezultati (test skup, 7.669 vijesti)

| Model | Accuracy | Precision (fake) | Recall (fake) | F1 (fake) | ROC-AUC |
|---|---|---|---|---|---|
| **Logistička regresija** | **98,40%** | 98,9% | 97,5% | 98,2% | **0,9985** |
| Random Forest | 96,49% | 98,3% | 93,7% | 96,0% | 0,9961 |
| Naive Bayes | 95,58% | 94,9% | 95,2% | 95,1% | 0,9901 |

Vrijednosti su izračunate iz matrica konfuzije na test skupu (`results/07_matrice_konfuzije.png`), a tabela je u `results/rezultati_modela.csv`.

**Glavni nalazi**
- Logistička regresija ima najbolje rezultate. Linearni modeli dobro rade na visokodimenzionalnim rijetkim TF-IDF matricama, a koeficijenti su interpretabilni.
- Random Forest ima visoku preciznost, ali propušta više lažnih vijesti (215 naspram 87 kod Logističke regresije).
- Greške modela: lažne vijesti pisane formalnim novinarskim stilom prolaze kao istinite, a istinite vijesti o kontroverznim temama sa emotivnim rječnikom označavaju se kao lažne. Model dakle uči **stil i vokabular**, a ne istinitost tvrdnji.

## Ograničenja

- Vijesti dolaze iz jasno razdvojenih izvora, pa klase mogu imati sistematske razlike u stilu, pa je visoka tačnost djelimično posljedica samog skupa. Na heterogenijim izvorima tačnost bi vjerovatno bila niža.
- Samo engleski jezik i uska tematika (američka politika 2015–2018).
- Jedna hold-out podjela, bez unakrsne validacije i bez sistematske optimizacije hiperparametara.
- Lematizator bez označavanja vrste riječi pravi greške (npr. *was* postaje *wa*).
- Na kratkim primjerima izvan skupa model je nesiguran: za realno zvučeću vijest o izvršnoj uredbi (2021+) vjerovatnoće su 50,1% fake naspram 49,9% real (`predict_news()` u bilježnici). Skup pokriva samo 2015–2018.
- Uklonjena je samo oznaka *Reuters*; druge formalne razlike u izvorima (npr. početak teksta sa nazivom grada) mogu i dalje biti signal za model.
- Sistem ne provjerava činjenice, već prepoznaje obrasce u tekstu, pa ne može zamijeniti ručnu provjeru.

## Moguća unapređenja

Naslov kao zaseban feature, word embeddings (Word2Vec, GloVe) i fine-tuning predtreniranih modela (npr. BERT).

## Struktura i pokretanje

```
data/        uputstvo za preuzimanje podataka
notebooks/   fake_news_detection.ipynb (kod sa sačuvanim izlazima)
results/     grafici i tabela rezultata
```

```bash
pip install -r requirements.txt
```

Preuzmi `Fake.csv` i `True.csv` u `data/` (vidi `data/README.md`), pa pokreni `jupyter notebook notebooks/fake_news_detection.ipynb`.
