# Deployment: api.aminskenderi.me via Cloudflare Tunnel

Das Backend läuft als Docker-Container auf dem eigenen Rechner (Laptop, später
Raspberry Pi) und wird über einen Cloudflare Tunnel als
`https://api.aminskenderi.me` veröffentlicht. Keine Portfreigabe nötig,
TLS macht Cloudflare.

Voraussetzung: Die Domain `aminskenderi.me` liegt als Zone in einem
Cloudflare-Account (Free Plan), Nameserver sind umgestellt.

## 1. Backend starten

```bash
cp .env.example .env   # einmalig, Keys eintragen
docker compose up -d --build
curl http://localhost:8000/health   # -> {"status":"ok"}
```

## 2. cloudflared installieren (Ubuntu/Debian, auch ARM64/Pi)

```bash
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/cloudflared.list
sudo apt update && sudo apt install cloudflared
```

## 3. Tunnel anlegen (einmalig)

```bash
cloudflared tunnel login                                # Browser-Login, Zone auswählen
cloudflared tunnel create pdf-api                       # merkt sich die Tunnel-ID
cloudflared tunnel route dns pdf-api api.aminskenderi.me
```

Dann `deploy/cloudflared/config.yml` nach `~/.cloudflared/config.yml` kopieren
und `<TUNNEL-ID>` durch die echte ID ersetzen (= Dateiname der erzeugten
`~/.cloudflared/*.json`).

Test im Vordergrund:

```bash
cloudflared tunnel run pdf-api
curl https://api.aminskenderi.me/health   # -> {"status":"ok"}
```

## 4. Als Dienst (Autostart)

```bash
sudo cloudflared service install   # kopiert Config nach /etc/cloudflared, legt systemd-Unit an
sudo systemctl enable --now cloudflared
systemctl status cloudflared
```

## 5. Umzug auf den Raspberry Pi

Der Tunnel hängt an der Credentials-Datei, nicht an der Maschine:

1. Auf dem Pi: Docker + cloudflared installieren (Schritte 1–2)
2. `~/.cloudflared/` (Config + `*.json`) vom Laptop auf den Pi kopieren,
   Pfad in der config.yml anpassen
3. Repo auf den Pi klonen, `.env` kopieren, `docker compose up -d --build`
4. `sudo cloudflared service install` — fertig, DNS/Frontend bleiben unverändert
5. Auf dem Laptop den cloudflared-Dienst deaktivieren (zwei aktive Verbindungen
   desselben Tunnels funktionieren zwar, sind hier aber nicht gewollt)

## Schutzmechanismen

- Rate Limit: `RATE_LIMIT_PER_MINUTE` (Default 10 POSTs/Minute pro Client-IP,
  erkannt via `CF-Connecting-IP`)
- Upload-Limit: `MAX_UPLOAD_BYTES` (Default 10 MB)
- CORS: `CORS_ORIGINS` (Default aminskenderi.me)
