# Audio Anomaly Detector

Reaaliaikainen äänianomalioiden tunnistusjärjestelmä Raspberry Pi:lle. Osa
[Drone_Detector](../README.md) -kokonaisuutta.

## Repository + Submodule

`rpi-lorawan` on Git-submodule (ks. juuren `.gitmodules`). Alusta se kloonauksen
jälkeen:

```bash
git submodule update --init --recursive
```

## Asennus

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Vaatii myös: `sudo apt install libportaudio2`

`scripts/`-hakemiston käynnistysskriptit (`detector-start` ym.) olettavat
virtuaaliympäristön nimeltä `.venv`.

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

Oletuksena monitori käyttää mallibundlea, jonka polku on asetettu tiedostossa
`config.py` (`DETECTOR_BACKEND = "bundle"`):

- `MODEL_BUNDLE_PATH` — nykyinen oletus:
  `Trained models/rf_private_real_16khz_verification_20260806/artifacts/v2_16khz/models/v2a/models/random_forest.joblib`
- `MODEL_SCALER_PATH` — vastaava scaler samasta artifacts-hakemistosta

Vanhemmat, yksinkertaisemmat bundlet (`Trained models/random_forest.joblib`,
`Trained models/tiny_mlp_es.joblib`) ovat yhä repossa ja käytettävissä
vaihtamalla nämä polut `config.py`:hin.

Voit muuttaa aggregointi-ikkunaa ajonaikaisella parametrilla, esim. 30 sekuntia:

```bash
python3 detector.py --aggregate-seconds 30
```

Tunnistukset logitetaan konsoliin ja `detections.log`-tiedostoon, ja jokainen
tunnistustapahtuma (ks. `DETECTION_WINDOW_CHUNKS` / `DETECTION_MIN_ANOMALIES_IN_WINDOW`
/ `DETECTION_COOLDOWN_SECONDS` -kynnykset `config.py`:ssä) lähetetään myös
[detector_ui](../detector_ui/)-web-käyttöliittymään `detector_client.py`:n
kautta (`DETECTOR_URL`/`CONFIDENCE_URL` ympäristömuuttujat, oletuksena
`http://127.0.0.1:8000`).

Jos `LORA_SEND_ENABLED = True`, tunnistin lähettää lisäksi LoRaWAN-uplinkin
`ANOMALY_REPORT_INTERVAL_SECONDS` sekunnin välein:
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

## Muut työkalut

- `probe_audio_device.py` — listaa äänilaitteet ja niiden natiivin
  näytteenottotaajuuden (käytä `MIC_SAMPLE_RATE`-asetuksen valintaan).
- `batch_test.py` (ajetaan yleensä `scripts/batch-test`-wrapperin kautta) —
  selainpohjainen työkalu mallin ajamiseen nauhoituksia vasten uudelleen
  ajettavasti, tulokset `batch_runs/`-hakemistoon.
- `analyze_logs.py`, `log_viewer.py` (`scripts/log-viewer`) — session-lokien
  (`logs/`) analysointi ja selainpohjainen katselu.
- `test_file.py` — ajaa tunnistimen yhtä wav-tiedostoa vasten
  (`scripts/detector-test` käyttää `test_samples/positive_test.wav`).
- `train_baseline.py` — kouluttaa yksinkertaisen IsolationForest-baselinen
  (`DETECTOR_BACKEND = "baseline"`).
