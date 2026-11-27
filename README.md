# Smart Home Energy Optimizer

Πρόβλεψη οικιακής κατανάλωσης ηλεκτρικών συσκευών και προσαρμοστική διαχείριση ενέργειας με μηχανική μάθηση.

**Πανεπιστήμιο Αιγαίου — ΠΜΣ «Διαδίκτυο των Πραγμάτων: Τεχνητή Νοημοσύνη και Σύγχρονα Δίκτυα»**  
Μάθημα: Συστήματα Διάχυτου Υπολογισμού | 2026–2027

## 1. Σκοπός

Το project αποτελεί proof-of-concept συστήματος διάχυτου υπολογισμού. Αξιοποιεί ιστορικές μετρήσεις αισθητήρων για πρόβλεψη της κατανάλωσης ηλεκτρικών συσκευών στο επόμενο δεκάλεπτο και αυτόματη επιλογή πολιτικής Normal/Eco. Η λειτουργία πραγματοποιείται σε εικονικό χρόνο, χωρίς πραγματικούς αισθητήρες ή ενεργοποιητές.

## 2. Εγκατάσταση και εκτέλεση

Το περιβάλλον ανάπτυξης και δοκιμών ήταν macOS με Python 3.13.0. Από τον κεντρικό φάκελο του project:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -m streamlit run app.py
```

Σε Windows, η ενεργοποίηση του virtual environment γίνεται με `.venv\Scripts\Activate.ps1` στο PowerShell. Η εγκατάσταση σε διαφορετικό λειτουργικό σύστημα δεν έχει επαληθευτεί. Το `requirements-lock.txt` καταγράφει το πλήρες περιβάλλον ανάπτυξης και δεν αποτελεί εγγύηση διαλειτουργικότητας.

Τα αποθηκευμένα μοντέλα και αποτελέσματα επιτρέπουν την εκτέλεση του dashboard χωρίς επανεκπαίδευση. Το αρχείο `energydata_complete.csv` πρέπει να βρίσκεται στον κεντρικό φάκελο.

## 3. Dataset και μεθοδολογία

Πηγή: [UCI Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction). Αν λείπει το CSV, κατεβάστε το dataset από την παραπάνω σελίδα και τοποθετήστε το `energydata_complete.csv` στον κεντρικό φάκελο.

- 19.735 εγγραφές, 29 στήλες, δειγματοληψία ανά 10 λεπτά.
- Περίοδος: 11/01/2016 17:00 έως 27/05/2016 18:00.
- Στόχος: `Appliances` (Wh) στη στιγμή t+10 λεπτά.
- Είσοδοι: ώρα, ημέρα εβδομάδας, T1, T2, T_out, RH_1, RH_2 και Appliances στις στιγμές t, t−10, t−20 λεπτά.
- Χρονολογικός διαχωρισμός 80/20 χωρίς shuffle, βάσει χρόνου στόχου. Μετά τη δημιουργία χαρακτηριστικών: 15.785 training και 3.947 test δείγματα.
- Όριο διαχωρισμού: 30/04/2016 08:20.
- Ρύθμιση υπερπαραμέτρων μόνο στο training set, με 100 τυχαίους συνδυασμούς, πέντε χρονικά διατεταγμένα folds και seed 42.

## 4. Αρχιτεκτονική

```mermaid
flowchart LR
    A[Dataset] --> B[Preprocessing]
    B --> C[Model Manager]
    C --> D[Prediction t+10]
    D --> E[Policy Engine]
    E --> F[Simulator]
    F --> G[Results and logs]
    G --> H[Streamlit Dashboard]
```

Ο προσομοιωτής επιλέγει πολιτική αποκλειστικά από την πρόβλεψη, πριν χρησιμοποιήσει την πραγματική τιμή του επόμενου διαστήματος για την αξιολόγηση της πολιτικής. Οι προσομοιωμένες τιμές δεν επιστρέφουν ως είσοδοι του μοντέλου.

## 5. Μοντέλα και αποτελέσματα πρόβλεψης

| Μοντέλο | Test MAE (Wh) | Test RMSE (Wh) |
|---|---:|---:|
| Persistence | 26,65 | 66,44 |
| Original XGBoost | 38,65 | 65,88 |
| Optimized XGBoost | 23,08 | 59,09 |
| Residual XGBoost | 23,38 | 59,34 |

Το persistence προβλέπει την τελευταία διαθέσιμη τιμή. Το Residual XGBoost προβλέπει τη μεταβολή από αυτήν και ανακατασκευάζει την τελική πρόβλεψη ως `max(0, Appliances(t) + predicted_residual)`.

Το Residual XGBoost επιλέχθηκε βάσει του χαμηλότερου training cross-validation MAE μεταξύ των υποψήφιων XGBoost μοντέλων (26,80 Wh, CV RMSE 65,27 Wh), όχι βάσει του χαμηλότερου test MAE. Το Optimized XGBoost εμφανίζει ελαφρώς καλύτερο test MAE. Το τελικό μοντέλο επιτυγχάνει F1 περίπου 78,45% για την ανίχνευση διαστημάτων άνω του ορίου Eco.

Τα αποτελέσματα αφορούν το συγκεκριμένο dataset. Τα ίδια CV folds χρησιμοποιήθηκαν για tuning και σύγκριση, ενώ το test set είχε εξεταστεί κατά την ανάπτυξη. Δεν υπάρχει ανεξάρτητη εξωτερική επικύρωση.

## 6. Ενεργειακή πολιτική

Το όριο Eco είναι το 75ο εκατοστημόριο της καταγεγραμμένης κατανάλωσης στο αρχικό training τμήμα: 100 Wh.

- Πρόβλεψη ≤ 100 Wh: Normal, χωρίς μείωση.
- Πρόβλεψη > 100 Wh: Eco, υποθετική μείωση 20% του υποθετικά ελεγχόμενου 30% του φορτίου.
- Υποθετική εξοικονόμηση σε Eco = πραγματική κατανάλωση × 0,30 × 0,20.

| Μετρική | Τελικό μοντέλο |
|---|---:|
| Διαστήματα | 3.947 |
| Normal / Eco | 3.020 / 927 |
| Eco (%) | 23,49% |
| Always Normal | 380.400 Wh |
| Adaptive Normal/Eco | 369.922,2 Wh |
| Υποθετική εξοικονόμηση | 10.477,8 Wh (2,75%) |
| Μέσος χρόνος επεξεργασίας πυρήνα | 1,440 ms |
| Μέγιστος χρόνος επεξεργασίας πυρήνα | 43,246 ms |

Οι χρόνοι αφορούν την καταγεγραμμένη εκτέλεση του πυρήνα, όχι τον χρόνο εκπαίδευσης, την ανανέωση του browser ή την τεχνητή καθυστέρηση αναπαραγωγής. Η εξοικονόμηση είναι υποθετική και δεν αποδεικνύει πραγματική μείωση ενέργειας. Περισσότερες ενεργοποιήσεις Eco αυξάνουν μηχανικά την υποθετική εξοικονόμηση και δεν συνεπάγονται καλύτερη πολιτική.

## 7. Dashboard και demo

- **Overview:** βασικές μετρικές και τελικά αποτελέσματα.
- **Live Simulation:** επιλογή μοντέλου, ταχύτητα εικονικού χρόνου, Start, Pause, Step, Reset, ιστορικό και εξαγωγή δεδομένων.
- **ML Evaluation:** σύγκριση μοντέλων, προβλέψεις, σφάλματα, ανίχνευση Eco και feature importance.
- **Energy Evaluation:** σύγκριση πολιτικών, αθροιστική κατανάλωση και καταγραφές.
- **System Architecture:** αρχιτεκτονική, παραδοχές και περιορισμοί.

Για demo μέσω terminal:

```bash
python simulator.py
python verify_integration.py
```

Η εκτέλεση του simulator μέσω terminal ανανεώνει τα αρχεία της προσομοίωσης. Το live dashboard δεν αντικαθιστά τα επίσημα αποτελέσματα.

## 8. Αναπαραγωγή πειραμάτων

Προαιρετικά, για νέα εκπαίδευση και παραγωγή αποτελεσμάτων (ενδέχεται να απαιτηθεί σημαντικός χρόνος):

```bash
python train_model.py
python evaluate_model.py
python tune_model.py
python train_residual_model.py
python simulator.py
python verify_integration.py
python evaluate_final.py
```

Οι εντολές εκπαίδευσης και αξιολόγησης μπορούν να αντικαταστήσουν υπάρχοντα artifacts. Κρατήστε αντίγραφο των `models/` και `results/` πριν τις εκτελέσετε. Η πλήρης ανεξάρτητη επανεκπαίδευση σε νέο περιβάλλον δεν έχει επαληθευτεί.

## 9. Testing και Quality Assurance

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests/ -v
python verify_integration.py
```

Εκτέλεση με coverage:

```bash
python -m pytest tests/ --cov=data_loader --cov=preprocessing --cov=policy_engine --cov=model_manager --cov=simulator --cov=app --cov-report=term-missing --cov-report=html:results/coverage_html --cov-report=json:results/coverage.json --junitxml=results/pytest_results.xml
```

| Κατηγορία | Επιτυχείς έλεγχοι |
|---|---:|
| Data Loader | 9/9 |
| Preprocessing | 14/14 |
| Policy Engine | 25/25 |
| Model Manager | 11/11 |
| Simulator | 13/13 |
| Integration | 9/9 |
| Dashboard | 10/10 |
| **Σύνολο** | **91/91** |

Επιπλέον: 11/11 integration verification checks, μηδενική διαφορά προβλέψεων έναντι των αποθηκευμένων αποτελεσμάτων και statement coverage 78% (575/741 εντολές σε έξι modules). Η εκτέλεση με coverage ολοκληρώθηκε σε 23,01 s και ο τελευταίος έλεγχος χωρίς coverage σε 15,01 s. Το coverage δεν αφορά το σύνολο των scripts εκπαίδευσης ούτε εγγυάται πλήρη κάλυψη όλων των συμπεριφορών του browser.

## 10. Δομή project

```text
.
├── .streamlit/config.toml
├── app.py
├── config.py
├── data_loader.py
├── preprocessing.py
├── train_model.py
├── tune_model.py
├── train_residual_model.py
├── model_manager.py
├── policy_engine.py
├── simulator.py
├── verify_integration.py
├── evaluate_model.py
├── evaluate_final.py
├── energydata_complete.csv
├── models/
├── results/
│   ├── final_figures/
│   ├── coverage_html/
│   ├── final_model_comparison.csv
│   ├── final_cv_comparison.csv
│   ├── model_selection.json
│   ├── energy_policy_comparison.csv
│   ├── simulation_log.csv
│   ├── simulation_metrics.json
│   ├── performance_metrics.json
│   └── pytest_results.xml
├── tests/
├── pytest.ini
├── requirements.txt
├── requirements-dev.txt
├── requirements-lock.txt
└── README.md
```

Η παραπάνω δομή είναι συνοπτική. Τα `models/` περιέχουν τα τρία εκπαιδευμένα XGBoost μοντέλα και τα metadata τους. Τα `results/` περιέχουν επίσης προβλέψεις, tuning results, γραφήματα και αναφορές ελέγχων.

## 11. Περιορισμοί

- Ιστορικά δεδομένα ενός χώρου, χωρίς ζωντανούς αισθητήρες ή πραγματικό έλεγχο συσκευών.
- Η μεταβλητή Appliances δεν αντιπροσωπεύει κατ’ ανάγκη τη συνολική κατανάλωση της κατοικίας.
- Τα ποσοστά ελεγχόμενου φορτίου και μείωσης αποτελούν παραδοχές, όχι μετρήσεις.
- Δεν αξιολογούνται πραγματική εξοικονόμηση, άνεση χρηστών ή γενίκευση σε άλλες κατοικίες.

## 12. Αναφορές

- [UCI Appliances Energy Prediction Dataset](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction).
- Candanedo, L. M., Feldheim, V., & Deramaix, D. (2017). Data driven prediction models of energy use of appliances in a low-energy house. *Energy and Buildings*, 140, 81–97. [DOI](https://doi.org/10.1016/j.enbuild.2017.01.083).
- [XGBoost documentation](https://xgboost.readthedocs.io/).
- [scikit-learn documentation](https://scikit-learn.org/stable/).
- [Streamlit documentation](https://docs.streamlit.io/).
