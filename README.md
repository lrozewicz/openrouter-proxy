# openrouter-proxy

Mały proxy zgodny z OpenAI API, który przed każdym zapytaniem wybiera najlepszego providera
OpenRouter dla danego modelu według reguł z `config.yaml` (cena, throughput, latency, uptime, kwantyzacja,
cache) i wysyła je do OpenRouter z odpowiednim `provider.order` / `provider.only`.

Dzięki temu proxy samo korzysta z promocji, póki trwają, i samo przełącza się na tańszego albo szybszego
providera, gdy promocja się skończy lub provider zwolni.

## Jak to działa

1. **Katalog providerów.** Co `refresh_interval` (np. 5m, 15m, 1h, 6h) proxy pobiera
   `GET /api/v1/models/{model}/endpoints`. Z kluczem API OpenRouter zwraca tam ceny (z promocjami
   i nadpisaniami godzinowymi lub dla długiego kontekstu), kwantyzację, uptime (5m/30m/1d)
   oraz throughput i latency jako percentyle p50–p99 z ostatnich 30 minut.
2. **Filtry.** Odrzucają providerów niespełniających progów. Zawsze sprawdzane jest też, czy prompt
   mieści się w kontekście providera i czy provider obsługuje `tools` oraz `response_format`, jeśli
   zapytanie ich używa.
3. **Hierarchia priorytetów.** Providerzy są porównywani po kolei według kryteriów z listy `priority`.
   Providerzy mieszczący się w `tolerance` od najlepszej wartości danego kryterium mają remis
   i rozstrzyga kolejne kryterium. Przykład: przy `tolerance: 0.15` na cenie wszyscy do 15% drożsi
   od najtańszego są równie dobrzy, więc wygrywa szybszy z nich.
4. **Wysyłka.** Do OpenRouter trafia `provider.order` (najlepsi `max_providers`) oraz `provider.only`
   (wszyscy, którzy przeszli filtry). Gdy najlepszy zwróci 429 lub 5xx, OpenRouter sam przechodzi do
   kolejnego, ale tylko w obrębie dozwolonych providerów.
5. **Pomiar.** Proxy obserwuje każdą odpowiedź (także streaming, bez opóźniania jej). Zapisuje, który
   provider faktycznie odpowiedział, `cached_tokens`, czas do pierwszego tokena i tok/s.

### Cache: dane od providera czy własny pomiar?

OpenRouter podaje tylko, czy provider wspiera prompt caching (`supports_implicit_caching`) i ile kosztuje
odczyt z cache. **Nie podaje, jak dobrze ten cache działa w praktyce**, więc proxy mierzy to samo:
`cached_tokens / prompt_tokens` dla kontynuacji rozmów z promptem powyżej 2k tokenów, liczone jako
średnia ruchoma (EWMA) per provider. Do czasu zebrania `min_cache_samples` pomiarów przyjmowana jest
wartość `expected_cache_hit_rate`.

Z tej wartości korzysta kryterium `effective_input_price`, czyli realna cena wejścia po uwzględnieniu
cache. Korzysta z niej także `blended_price`, które dokłada do tego cenę wyjścia. Przy programowaniu
to zwykle najlepsze kryterium cenowe.

### Próg kontekstu (`switch_below_context`)

Zmiana providera w środku sesji oznacza utratę cache, a przy dużym kontekście to boli. Dlatego:

- **poniżej progu** (np. `100k`) proxy przełącza się swobodnie na najlepszego providera;
- **powyżej progu** zachowanie ustawia `above_context`:
  - `sticky` (domyślnie): zostaje przy providerze, którego ta sama sesja używała wcześniej, o ile
    nadal przechodzi filtry. Sesję rozpoznaje po prompcie systemowym i pierwszej wiadomości
    użytkownika albo po nagłówku `X-Session-Id`;
  - `ranked`: zawsze bierze najlepszego providera;
  - `openrouter`: używa domyślnego routingu OpenRouter.

### ZDR

Jeśli konto OpenRouter ma włączone Zero Data Retention, OpenRouter odrzuca część providerów,
np. Baidu, StreamLake i Alibaba dla DeepSeek V4 Flash. Ustaw wtedy `zdr_only: true`. Proxy pobierze
listę endpointów ZDR (`/api/v1/endpoints/zdr`) i nie będzie wybierać providerów, których OpenRouter
i tak by nie użył.

## Konfiguracja

Pełny, skomentowany przykład jest w [`config.yaml`](config.yaml). Konfigurację proxy czyta ze zmiennej
`ORPROXY_CONFIG_YAML`, a gdy jest pusta, z pliku `ORPROXY_CONFIG` (domyślnie `config.yaml`, w formacie
YAML albo JSON z rozszerzeniem `.json`).

| Filtr | Znaczenie |
|---|---|
| `min_throughput` | minimalny throughput, tok/s (percentyl wybrany w `metrics_percentile`) |
| `max_latency` | maksymalny czas do pierwszego tokena, ms |
| `max_input_price` / `max_output_price` / `max_cache_read_price` | maksymalna cena, $ za 1M tokenów |
| `min_uptime` | minimalny uptime w %, w oknie `uptime_window` (5m / 30m / 1d) |
| `quantizations` / `min_quantization` / `allow_unknown_quantization` | jakość modelu (kwantyzacja) |
| `require_cache` / `min_cache_hit_rate` | wsparcie cache / zmierzony cache hit |
| `zdr_only` | tylko endpointy Zero Data Retention |
| `include_providers` / `exclude_providers` | biała i czarna lista (slug, tag lub nazwa providera) |
| `strict_metrics` | `true`: brak danych o metryce oznacza odrzucenie (domyślnie brak danych przepuszcza) |

Każdy model wymaga co najmniej jednego filtra.

Kryteria `priority`: `effective_input_price`, `blended_price`, `input_price`, `output_price`,
`cache_read_price`, `throughput`, `latency`, `uptime`, `cache_hit_rate`, `quantization`, `discount`,
`context_length`. Każde można podać jako sam string albo jako `{by: ..., tolerance: 0.2}`.

`on_no_match` określa, co zrobić, gdy nikt nie przejdzie filtrów:

- `best_effort`: ranking wszystkich zgodnych providerów;
- `openrouter`: domyślny routing OpenRouter;
- `error`: zwraca HTTP 503.

Modele spoza listy są przekazywane dalej bez zmian, jeśli `passthrough_unknown_models: true`.

## Endpointy

- `POST /v1/chat/completions`, `/v1/completions` i każde inne `/v1/*` jest przekazywane do OpenRouter.
  Routing jest wstrzykiwany, gdy `model` to alias z konfiguracji albo identyfikator OpenRouter skonfigurowanego
  modelu (np. `deepseek/deepseek-v4-flash-0731`). Obie formy mogą mieć prefiks `openrouter/`.
- `GET /v1/models` zwraca skonfigurowane aliasy.
- `GET /status?tokens=50000` (wymaga klucza) pokazuje aktualny ranking, powód odrzucenia każdego
  providera, jego metryki i pomiary proxy. Parametr `tokens` symuluje rozmiar promptu.
- `GET /health` służy do healthchecku.

Nagłówki odpowiedzi `x-orproxy-mode` i `x-orproxy-order` pokazują, jaką decyzję podjęło proxy.

## Wdrożenie (Portainer / Swarm)

Obraz `ghcr.io/lrozewicz/openrouter-proxy` buduje GitHub Actions (`.github/workflows/docker.yml`):

- na każdym PR do `main` uruchamiane są testy i próbna budowa obrazu, bez publikacji;
- po merge do `main` obraz jest publikowany z tagami `latest` oraz `sha-<commit>`;
- tag `v*` publikuje dodatkowo obraz z tym tagiem.

W Portainerze:

1. **Secrets → Add secret**:
   - `openrouter_api_key`: klucz OpenRouter `sk-or-...`;
   - `openrouter_proxy_api_keys`: klucze dla klientów proxy, np. wynik `openssl rand -hex 32`,
     jeden na linię.
2. **Stacks → Add stack → Web editor**: wklej zawartość [`docker-compose.yml`](docker-compose.yml)
   i kliknij Deploy. Traefik wystawi usługę pod `https://openrouter-proxy.lucaswebart.com`.

Cała konfiguracja routingu jest w stacku, w zmiennej `ORPROXY_CONFIG_YAML`, w tym samym formacie
co [`config.yaml`](config.yaml). Żeby ją zmienić, edytuj stack i kliknij „Update the stack”.
W treści konfiguracji nie używaj znaku `$`, bo compose traktuje go jako zmienną.

Nowy obraz po merge do `main` wdrożysz przez „Update the stack” z zaznaczonym
„Re-pull image and redeploy”.

Lokalnie, bez Swarm:

```bash
pip install -r requirements.txt
OPENROUTER_API_KEY=sk-or-... python -m orproxy config.yaml
```

Do lokalnego uruchomienia ustaw w configu `auth.api_keys: [dev]` albo `auth.allow_anonymous: true`.

Testy: `pip install pytest && pytest`.

## OpenCode

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "orproxy": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "OpenRouter Proxy",
      "options": {
        "baseURL": "https://openrouter-proxy.lucaswebart.com/v1",
        "apiKey": "{env:ORPROXY_API_KEY}"
      },
      "models": {
        "deepseek-v4-flash": { "name": "DeepSeek V4 Flash (auto-provider)" }
      }
    }
  }
}
```

## Ograniczenia

- Liczba tokenów promptu jest szacowana z długości tekstu, bez tokenizera. Proxy kalibruje współczynnik znaków
  na token osobno dla każdego modelu, porównując szacunek z faktycznym `prompt_tokens` z odpowiedzi (prompty
  od 1000 tokenów, po 3 próbkach). Do tego czasu używa `chars_per_token`. Aktualną wartość pokazuje `/status`.
- Stan pomiarów (cache hit, tok/s) i sesje sticky są trzymane w pamięci i giną przy restarcie. Zbierają
  się ponownie w ciągu kilku zapytań. Uruchamiaj jedną replikę.
- Inne ograniczenia konta OpenRouter (np. zablokowani providerzy, polityka `data_collection`) nie są
  widoczne w API endpointów. Jeśli coś wycinają, dodaj tych providerów do `exclude_providers`.
