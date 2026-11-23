# Assistente de Suporte Técnico com IA e Engenharia de Prompt

Aplicação modular em Python desenvolvida para o **Desafio Técnico — IA e Engenharia de Prompt**. O sistema recebe um **perfil de cliente** estruturado segundo a matriz ortogonal **3 × 3 × 3** (`technical_level` × `communication_style` × `channel`) e um **ticket de suporte técnico**, gerando **4 artefatos especializados** via motor de prompts versionado (`v1` vs `v2`), protegido contra *Prompt Injection* (*XML Tag Breakout*), instrumentado com **telemetria de latência (`latency_ms`)**, **cache determinístico SHA-256**, **persistência atômica em JSON**, **CLI em `rich`** e **console web interativo em Dark Mode com Streamlit + Altair**.

---

## 1. Visão Geral e Objetivos

Equipes de suporte técnico enfrentam alta variabilidade tanto nos incidentes relatados quanto no perfil dos clientes — de um lojista leigo falando ao telefone a um engenheiro sênior abrindo um ticket técnico via chat. Chamadas genéricas a LLMs geram respostas inconsistentes, vazamento de jargão, risco de *prompt injection* e custos repetidos sem auditabilidade.

O projeto resolve esse problema por meio de uma **Arquitetura Modular Orientada a Pipeline** em 4 camadas desacopladas:

- **Calibração Ortogonal (3 × 3 × 3 = 27 combinações com apenas 9 regras):**
  - `technical_level`: `leigo`, `intermediário`, `avançado`
  - `communication_style`: `direto`, `detalhado`, `didático`
  - `channel`: `chat`, `e-mail`, `telefone`
- **4 Estratégias de Artefato Separadas (`ContentType`):**
  1. `diagnosis` — Diagnóstico estruturado em **Hipóteses**, **Evidências Disponíveis**, **Dados Faltantes** e **Conclusão Provável** (com nível de confiança e sem *Chain-of-Thought* bruto).
  2. `resolution` — Passo a passo numerado, calibrado ao perfil/canal, com ação concreta, sinal esperado de sucesso e protocolo de escalonamento para N2.
  3. `clarification` — 3 a 5 perguntas essenciais priorizadas com justificativa de impacto diagnóstico.
  4. `flow_summary` — Fluxograma ASCII padronizado (`+`, `-`, `|`, `>`, `v`, `[SIM]`/`[NÃO]`) com preservação integral de indentação.
- **Reprodutibilidade e Governança:** Versionamento explícito (`v1` Baseline vs `v2` Otimizada), cache SHA-256 canônico, telemetria de latência (`latency_ms`), rubrica obrigatória de 3 critérios (`clarity`, `customer_fit`, `actionability` na escala 1–5, total `/15`) e **123 testes automatizados 100% offline**.

---

## 2. Arquitetura e Estrutura de Diretórios

### Diagrama do Pipeline Ponta a Ponta

```text
+-------------------------------------------------------------------------+
|                        INTERFACES DE USUÁRIO                            |
|  CLI (cli.py / src/cli.py - rich) | Web Dark Mode (src/web_app.py)      |
+-----------------------------------+-------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                     CONTRATOS PYDANTIC v2 E DADOS                       |
|   data/profiles.json   |   data/tickets.json   |   src/models.py        |
+-----------------------------------+-------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|               ORQUESTRADOR DO PIPELINE (src/pipeline/runner.py)         |
|                                                                         |
|   1. Hash SHA-256 canônico (src/pipeline/cache.py)                      |
|      customer_id + ticket + content_type + version + model + temp       |
|                                   |                                     |
|                                   v                                     |
|                        +---------------------+                          |
|                        | Existe no histórico |                          |
|                        | e !force_refresh?   |                          |
|                        +----+-----------+----+                          |
|                             |           |                               |
|                          [SIM]        [NÃO]                             |
|                             |           |                               |
|                             v           v                               |
|                +-----------------+ +----------------------------------+ |
|                | CACHE HIT       | | 2. Motor de Prompts Offline      | |
|                | (< 1 ms, 0 API) | |    src/prompts/strategies.py     | |
|                | elapsed_ms via  | |    src/prompts/engine.py         | |
|                | perf_counter()  | |    sanitize_ticket_text()        | |
|                +--------+--------+ |    versions/v1.py | v2.py        | |
|                         |          +----------------+-----------------+ |
|                         |                           |                   |
|                         |                           v                   |
|                         |          +----------------------------------+ |
|                         |          | 3. Cliente Desacoplado Gemini    | |
|                         |          |    src/services/llm_client.py    | |
|                         |          |    BaseLLMClient / GeminiClient  | |
|                         |          |    + medição latency_ms (ms)     | |
|                         |          +----------------+-----------------+ |
|                         |                           |                   |
|                         |                           v                   |
|                         |          +----------------------------------+ |
|                         |          | 4. Persistência Atômica em JSON  | |
|                         |          |    src/services/storage.py       | |
|                         |          |    .tmp -> replace()             | |
|                         |          |    -> data/history.json          | |
|                         |          +----------------+-----------------+ |
|                         |                           |                   |
|                         +-------------+-------------+                   |
|                                       |                                 |
|                                       v                                 |
|                      +--------------------------------+                 |
|                      | 5. Rubrica & Dashboard (1-5)   |                 |
|                      |    src/pipeline/evaluator.py   |                 |
|                      |    Altair Analytics + Drilldown|                 |
|                      +--------------------------------+                 |
+-------------------------------------------------------------------------+
```

### Estrutura Atualizada do Repositório

```text
.
├── .streamlit/
│   └── config.toml                           # Tema nativo Dark Mode técnico (#0E1117 / #161B22 / #38BDF8)
├── cli.py                                    # Atalho na raiz delegando para src.cli:main()
├── requirements.txt                          # 6 dependências mínimas compatíveis
├── .env.example                              # Template seguro de variáveis de ambiente (sem segredos)
├── .gitignore                                # Ignora .venv/, caches, .env e data/history.json
├── README.md                                 # Documentação principal de arquitetura, uso e qualidade
├── PROMPT_ENGINEERING_NOTES.md               # Análise aprofundada de prompts (v1 vs v2) e rubrica
├── RELATORIO_TECNICO_COMPLETO.pdf            # Relatório técnico consolidado em PDF
├── data/
│   ├── profiles.json                         # 4 perfis contrastantes cobrindo todas as categorias
│   └── tickets.json                          # 4 tickets realistas de suporte técnico
├── samples/                                  # Amostras reais versionadas e avaliadas com rubrica
│   ├── comparison_v1_vs_v2_diagnosis.json    # Comparação controlada v1 (8/15) vs v2 (14/15) no TKT-102
│   ├── sample_resolution_advanced.json       # Resolução v2 para perfil avançado no chat (TKT-103, 15/15)
│   ├── sample_clarification_layman.json      # Esclarecimento v2 para perfil leigo por telefone (TKT-104, 15/15)
│   └── sample_flow_summary_ascii.json        # Fluxograma ASCII v2 preservado (TKT-101, 14/15)
├── scripts/
│   └── generate_samples.py                   # Script reproduzível com retry exponencial para samples/
├── src/
│   ├── __init__.py
│   ├── models.py                             # Contratos Pydantic v2 (CustomerProfile, ExecutionRecord + latency_ms)
│   ├── cli.py                                # Interface CLI (modo interativo + flags argparse + latência)
│   ├── web_app.py                            # Console Web Dark Mode (3 abas, Altair analytics e drill-down)
│   ├── prompts/
│   │   ├── __init__.py                       # Fachada pública do motor de prompts
│   │   ├── engine.py                         # build_prompt(...), sanitize_ticket_text(...) e NO_RAW_COT_RULE
│   │   ├── strategies.py                     # 4 estratégias separadas + get_prompt_for_content_type
│   │   └── versions/
│   │       ├── __init__.py                   # Registro de versões (v1, v2) com validação estrita
│   │       ├── v1.py                         # Baseline minimalista
│   │       └── v2.py                         # Otimizada com matriz ortogonal 3x3x3 e contratos rígidos
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm_client.py                     # Interface BaseLLMClient + GeminiClient (google-genai SDK)
│   │   └── storage.py                        # JsonHistoryStorage com escrita atômica (.tmp + replace)
│   └── pipeline/
│       ├── __init__.py
│       ├── cache.py                          # generate_cache_key (SHA-256) e find_cached_execution
│       ├── evaluator.py                      # build_rubric_evaluation e evaluate_execution
│       └── runner.py                         # execute_pipeline + telemetria elapsed_ms/latency_ms
└── tests/
    ├── test_stage1_setup.py                  # Fundação, fixtures JSON e limites da rubrica (5 testes)
    ├── test_stage2_prompts.py                # Motor offline, 4 pilares e evolução v1 vs v2 (5 testes)
    ├── test_stage3_pipeline.py               # Cache, storage atômico, mocks e telemetria latency_ms (7 testes)
    ├── test_stage4_interfaces.py             # CLI, helpers web, rubrica analítica e dashboard (10 testes)
    ├── test_stage5_deliverables.py           # Integridade de samples/, docs e auditoria de segredos (5 testes)
    └── test_security_prompt_injection.py     # Suíte adversarial de Prompt Injection e XML Breakout (91 testes)
```

---

## 3. Instalação, Configuração e Execução

### Pré-requisitos
- **Python 3.10+** (compatível com Windows PowerShell e Linux/macOS).
- Chave válida da **Google Gemini API** (Google AI Studio).

### Passo 1 — Criar Ambiente Virtual e Instalar Dependências

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Passo 2 — Configurar Variáveis de Ambiente (`.env`)

Copie `.env.example` para `.env` e informe sua chave (o arquivo `.env` permanece ignorado pelo `.gitignore`):

```powershell
Copy-Item .env.example .env
```

```dotenv
GEMINI_API_KEY=sua_chave_api_aqui
LLM_PROVIDER=google
LLM_MODEL=gemini-2.5-flash
LLM_TEMPERATURE=0.2
```

### Passo 3 — Executar a CLI (`rich`)

Funciona tanto via `python cli.py` quanto via `python -m src.cli`:

```powershell
# Modo interativo guiado (tabelas numeradas de perfis, tickets, tipos e versões)
python cli.py

# Modo direto com exibição do prompt montado
python cli.py --customer cust_001 --ticket-id TKT-102 --type diagnosis --version v2 --show-prompt

# Modo direto com relato livre e avaliação da rubrica via flags
python cli.py `
  --customer cust_003 `
  --ticket-text "Endpoint /v1/orders retorna HTTP 502 intermitente após deploy." `
  --type resolution --version v2 `
  --clarity 5 --clarity-justification "Passo a passo objetivo e verificável." `
  --customer-fit 5 --customer-fit-justification "Vocabulário técnico adequado ao perfil avançado." `
  --actionability 5 --actionability-justification "Ações claras com evidências para escalonamento N2."
```

> Execute o mesmo comando duas vezes para observar o `CACHE HIT` instantâneo (sem chamada à API); passe `--force-refresh` para ignorar o cache.

### Passo 4 — Executar o Console Web (Streamlit Dark Mode)

```powershell
streamlit run src/web_app.py
```

### Passo 5 — Regenerar Amostras Versionáveis e Rodar Testes

```powershell
# Regenera/audita samples/ (reaproveita cache por padrão; use --force-refresh para nova chamada API)
python scripts/generate_samples.py

# Executa toda a suíte offline (123 testes)
pytest -v
```

---

## 4. Engenharia de Prompt e Evolução Controlada (`v1` vs `v2`)

O motor `src/prompts/engine.py` monta cada prompt em **6 blocos modulares** (`[PROMPT_METADATA]`, `[PERSONA E LIMITES]`, `[CONTEXTO DO CLIENTE E DO TICKET]`, `[ADAPTAÇÃO AO PERFIL E CANAL]`, `[TAREFA E DECOMPOSIÇÃO]` e `[FORMATO DE SAÍDA E RESTRIÇÕES]`):

- **Isolamento de Contexto:** O relato do cliente é sanitizado por `sanitize_ticket_text(...)` e isolado dentro de uma única tag XML `<ticket>...</ticket>`, acompanhado de regra explícita na `v2` tratando o conteúdo estritamente como dado.
- **Ancoragem Factual e Supressão de CoT Bruto:** Limites explícitos impedem o modelo de inventar logs, métricas ou SLAs; a cláusula transversal `NO_RAW_COT_RULE` proíbe exibir raciocínio interno bruto (`<thinking>` / *Chain-of-Thought* não filtrado).
- **Calibração Ortogonal (3 × 3 × 3):** Em `src/prompts/versions/v2.py`, `TECHNICAL_LEVEL_RULES`, `CHANNEL_RULES` e `COMMUNICATION_STYLE_RULES` compõem dinamicamente as instruções para todas as 27 combinações sem duplicar templates.

### Resultados Empíricos da Rubrica (`samples/*.json`, `google/gemini-3.5-flash-lite`, `temp=0.2`)

| Cenário (`samples/`) | Perfil / Canal | Artefato | Versão | Clareza | Adequação | Acionabilidade | Total |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `comparison_v1_vs_v2_diagnosis.json` (`TKT-102`) | `cust_001` (`leigo` / `didático` / `chat`) | `diagnosis` | **`v1`** | 3 / 5 | 2 / 5 | 3 / 5 | **8 / 15** |
| `comparison_v1_vs_v2_diagnosis.json` (`TKT-102`) | `cust_001` (`leigo` / `didático` / `chat`) | `diagnosis` | **`v2`** | 4 / 5 | 5 / 5 | 5 / 5 | **14 / 15 (+75%)** |
| `sample_resolution_advanced.json` (`TKT-103`) | `cust_003` (`avançado` / `direto` / `chat`) | `resolution` | **`v2`** | 5 / 5 | 5 / 5 | 5 / 5 | **15 / 15** |
| `sample_clarification_layman.json` (`TKT-104`) | `cust_004` (`leigo` / `direto` / `telefone`) | `clarification` | **`v2`** | 5 / 5 | 5 / 5 | 5 / 5 | **15 / 15** |
| `sample_flow_summary_ascii.json` (`TKT-101`) | `cust_002` (`intermediário` / `detalhado` / `e-mail`) | `flow_summary` | **`v2`** | 5 / 5 | 4 / 5 | 5 / 5 | **14 / 15** |

> A análise detalhada (incluindo o diagnóstico honesto do eco de instruções nos cabeçalhos do `diagnosis` em modelos *lite* e o plano de refinamento para uma futura `v3`) está documentada em [`PROMPT_ENGINEERING_NOTES.md`](PROMPT_ENGINEERING_NOTES.md).

---

## 5. Diferenciais Implementados (Observabilidade, Dashboard e Segurança)

### Console Técnico em Dark Mode (`.streamlit/config.toml` + `src/web_app.py`)
- Tema nativo escuro de alto conforto visual (`#0E1117`, `#161B22`, `#E6EDF3`, acento `#38BDF8`) com CSS leve para badges/pills de perfil e banners de status.
- **Aba 1 (`Gerar Atendimento`):** Workspace dividido em 2 colunas (`st.columns([1, 1.35])`), separando configuração (esquerda) de saída, inspeção do prompt e formulário da rubrica (direita).
- **Aba 2 (`Comparação Controlada v1 vs v2`):** Execução justa lado a lado compartilhando a mesma instância de cliente LLM (mesmo modelo e temperatura) com formulários individuais de rubrica.

### Telemetria de Latência (`latency_ms` & `elapsed_ms`)
- Instrumentação via `time.perf_counter()` em `src/pipeline/runner.py` sem alterar o cálculo do hash SHA-256 (`cache_key`).
- Em **novas execuções API**, o tempo da chamada ao provedor é persistido em `ExecutionRecord.latency_ms` (arredondado em 2 casas decimais) e exibido no banner (`✨ NOVA EXECUÇÃO API · X.XX ms`).
- Em **Cache Hits**, o tempo de leitura local (`elapsed_ms`, tipicamente `< 1 ms`) é exibido junto à latência original da API e à economia percentual estimada (`⚡ CACHE HIT · X.XX ms · economia de ~99.x%`), mantendo `record.latency_ms` intacto em `data/history.json`.
- Registros legados sem `latency_ms` carregam normalmente como `None` (`— (registro legado)`).

### Dashboard Interativo de Análise de Tickets (Aba 3 — `Histórico e Auditoria`)
- Cruza `data/history.json` com `data/profiles.json` via funções puras offline (`extract_ticket_identifier`, `build_ticket_analytics_rows`, `summarize_by_ticket`, `aggregate_rubric_metrics`).
- **Filtros Globais Sincronizados + Exportação:** Filtra KPIs, gráficos e tabela simultaneamente por *Ticket*, *Nível Técnico*, *Canal*, *Tipo de Artefato*, *Versão* e *Somente avaliadas*, com botões **Baixar JSON Filtrado** e **Baixar CSV Analítico** (incluindo `latency_ms`).
- **4 KPIs Dinâmicos:** *Execuções no Recorte*, *Nota Média Geral (`/15`)* com delta `v2 vs v1`, *Melhor Dimensão* e *Ticket Mais Crítico*.
- **Grid 2 × 2 de Gráficos Altair em Dark Mode:**
  1. *Desempenho por Ticket (`v1` vs `v2`)* — barras horizontais agrupadas na escala `[0, 15]`.
  2. *Heatmap `Nível Técnico × Canal`* — `mark_rect` + `mark_text` alternável entre `Total (/15)` e `Adequação (/5)`.
  3. *Comparativo por Critério da Rubrica (`v1` vs `v2`)* — barras agrupadas na escala fixa `[0, 5]`.
  4. *Distribuição por Tipo de Artefato (`content_type`)* — volume × nota média (`/15`).
- **Explorador com *Drill-Down*:** `st.dataframe(..., on_select="rerun", selection_mode="single-row")` acoplado ao **Painel Inspetor do Ticket Selecionado** (badges de perfil, metadados + latência, enunciado completo, 3 cards da rubrica e resposta completa).

### Segurança contra *Prompt Injection*
- **Sanitização contra *XML Tag Breakout*:** `sanitize_ticket_text(...)` em `src/prompts/engine.py` escapa qualquer variação *case-insensitive* de `<ticket>` / `</ticket>` injetada pelo usuário (ex.: `</TICKET>`, `</ ticket>`, `<ticket role="system">`), sem modificar um único byte de tickets legítimos nem quebrar hashes SHA-256.
- **Suíte Adversarial (`tests/test_security_prompt_injection.py`):** 91 testes automatizados cobrindo 6 vetores de ataque (*Instruction Override*, *Role Hijacking*, *XML Delimiter Breakout*, *Calibration Subversion*, *Factual Grounding Bait* e *Multiline Markdown Injection*), todas as 27 combinações da matriz ortogonal 3 × 3 × 3 e execução ponta a ponta no pipeline com `SpyLLMClient`.

---

## 6. Qualidade e Testes Automatizados (`123 passed`)

Toda a suíte executa **100% offline** (~1,1 s), sem consumir créditos de API e sem poluir `data/history.json`:

| Arquivo de Teste | Casos | Escopo Validado |
| :--- | :---: | :--- |
| `tests/test_stage1_setup.py` | 5 | Estrutura raiz, validação Pydantic de `profiles.json` e `tickets.json`, limites 1–5 da rubrica e schema `ExecutionRecord`. |
| `tests/test_stage2_prompts.py` | 5 | 4 estratégias em `v1`/`v2`, presença de perfil/ticket/versão/anti-CoT, 4 pilares de `diagnosis`, diff `v1` vs `v2` e `ValueError`. |
| `tests/test_stage3_pipeline.py` | 7 | Determinismo SHA-256, *cache miss/hit*, `force_refresh`, rubrica + preservação ASCII, validação `GEMINI_API_KEY`, legado sem `latency_ms` e telemetria `latency_ms`. |
| `tests/test_stage4_interfaces.py` | 10 | CLI não-interativa + rubrica + tratamento de erros, helpers web offline, `aggregate_rubric_metrics`, `extract_ticket_identifier`, `build_ticket_analytics_rows` (+ `latency_ms`/CSV) e `summarize_by_ticket`. |
| `tests/test_stage5_deliverables.py` | 5 | Validação Pydantic dos 4 arquivos de `samples/`, equidade da comparação `v1` vs `v2`, integridade de `README.md` / `PROMPT_ENGINEERING_NOTES.md` e varredura anti-vazamento de chaves `AIza...`. |
| `tests/test_security_prompt_injection.py` | 91 | Sanitização XML, retrocompatibilidade byte-a-byte, invariantes estruturais (6 vetores × 2 versões × 4 artefatos), invariância das 27 combinações ortogonais e pipeline ponta a ponta com `SpyLLMClient`. |
| **Total** | **123** | **100% aprovados (`pytest -v`), com importação tardia verificada (`streamlit`, `altair` e `pandas` não carregados ao importar `src.web_app`).** |

---

## 7. Decisões Técnicas, Trade-offs e Documentação Complementar

- **Desacoplamento de Provedor (`BaseLLMClient`):** Permite testes determinísticos offline (`FakeLLMClient` / `SpyLLMClient`) e troca de provedor sem alterar o pipeline.
- **Preservação Exata de Fluxogramas ASCII:** `ExecutionRecord` não aplica `str_strip_whitespace=True`; a CLI renderiza com `rich.text.Text` e o Streamlit com `st.code(..., language="text")`.
- **Escrita Atômica:** `JsonHistoryStorage` grava em `.tmp` e substitui via `Path.replace()`, prevenindo corrupção do histórico JSON.
- **Documentação Detalhada:** Consulte [`PROMPT_ENGINEERING_NOTES.md`](PROMPT_ENGINEERING_NOTES.md) para aprofundamento em engenharia de prompt e [`RELATORIO_TECNICO_COMPLETO.pdf`](RELATORIO_TECNICO_COMPLETO.pdf) para o relatório executivo e técnico completo do projeto.
