# Audio Anomaly Detector

Reaaliaikainen äänianomalioiden tunnistusjärjestelmä Raspberry Pi:lle.

## Repository + Submodule

`rpi-lorawan` on lisatty Git-submoduleksi.

Kloonaa projekti submoduleineen:

```bash
git clone --recurse-submodules git@github.com:cinetcampus/drone_detect.git
```

Jos olet jo kloonannut ilman submoduleja:

```bash
git submodule update --init --recursive
```

## Asennus

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Vaatii myös: `sudo apt install libportaudio2`

## Käyttö

### 1. Opi normaali taustaääni (baseline)

```bash
source venv/bin/activate
python3 train_baseline.py
```

Tämä nauhoittaa 30s normaalia ääntä ja tallentaa mallin. Voit muuttaa kestoa:
```bash
python3 train_baseline.py 60  # 60 sekuntia
```

### 2. Käynnistä monitorointi

```bash
python3 detector.py
```

Oletuksena monitori käyttää mallibundlea hakemistosta `Trained models/`
(`DETECTOR_BACKEND = "bundle"` tiedostossa `config.py`).

Mallibundle käyttää:
- `Trained models/random_forest.joblib`
- `Trained models/v2_spectral_mfcc_scaler.json`


Voit muuttaa aggregointi-ikkunaa ajonaikaisella parametrilla, esim. 30 sekuntia:

```bash
python3 detector.py --aggregate-seconds 30
```

Tunnistukset logitetaan konsoliin ja `detections.log`-tiedostoon.
Lisäksi tunnistin lähettää LoRaWAN-uplinkin kerran minuutissa:
- payload: Measurement-paketti (measurement.cpp/h formaatti)
- kenttä: `Observations` + `TimeSeries`
- sisältö: aggregointi-ikkunan per-chunk ennusteet (0/1), missä 1 = anomalia
- rakennne: `[header][length][type][count][predictions...]`

LoRa-asetukset löytyvät tiedostosta `config.py` (OTAA-avaimet, pinniasetukset,
binary-polut ja `ANOMALY_REPORT_INTERVAL_SECONDS`).

## Konfiguraatio

Muokkaa `config.py`:
- `THRESHOLD_MULTIPLIER` — herkkyys (pienempi = herkempi)
- `CHUNK_DURATION` — analyysi-ikkunan kesto
- `BASELINE_DURATION` — baseline-nauhoituksen pituus
