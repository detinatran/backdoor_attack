# RF Backdoor — Protocol-Agnostic Backdoor on RF Fingerprinting

Reproduction of: *"Backdoor Attack on RF Fingerprinting"* (INFOCOM 2025)

## Attack
- **Trigger**: short fixed I/Q pattern injected into RF signal
- **Goal**: spoof device identity across protocols (WiFi, BLE, Zigbee)
- **Model**: CNN on I/Q signals

## Results
| Metric | Value |
|--------|-------|
| Benign ACC | 49.03% |
| ASR | ~92% |
| Platform | NVIDIA A5000 |
