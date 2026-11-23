# Assistente de Suporte Técnico com IA e Engenharia de Prompt

Aplicação modular em Python desenvolvida para o **Desafio Técnico — IA e Engenharia de Prompt**. O sistema recebe um **perfil de cliente** (nível técnico, estilo de comunicação e canal de atendimento) e um **ticket de suporte**, gera quatro artefatos especializados via engenharia de prompt versionada (`v1` vs `v2`), aplica **cache determinístico SHA-256**, persiste o histórico validado em **JSON** e permite avaliação comparativa controlada com a **rubrica obrigatória de 3 critérios**.

---

## Visão Geral e Problema Resolvido

Equipes de suporte técnico frequentemente entregam respostas excessivamente técnicas para usuários leigos — ou rasas demais para desenvolvedores — além de desperdiçar chamadas repetidas a APIs de LLM sem controle de versão de prompt.

Este projeto resolve o problema por meio de uma **Arquitetura Modular Orientada a Pipeline** que:

- Adapta vocabulário, profundidade e formatação dinamicamente ao perfil (`leigo`, `intermediário`, `avançado`), estilo (`direto`, `detalhado`, `didático`) e canal (`chat`, `e-mail`, `telefone`).
- Gera os **4 tipos de conteúdo obrigatórios** por meio de estratégias separadas:
  1. `diagnosis` — Diagnóstico estruturado em Hipóteses, Evidências Disponíveis, Dados Faltantes e Conclusão Provável (sem Chain-of-Thought bruto).
  2. `resolution` — Passo a passo acionável com critérios de validação e escalonamento N2.
  3. `clarification` — 3 a 5 perguntas priorizadas com justificativa diagnóstica.
  4. `flow_summary` — Fluxograma ASCII limpo (`+`, `-`, `|`, `>`, `v` com `[SIM]`/`[NÃO]`).
- Evita chamadas repetidas via cache local determinístico (SHA-256 sobre parâmetros canônicos).
- Oferece **CLI obrigatória com `rich`** (interativa e por flags) e **Interface Web interativa em Streamlit** (incluindo comparação lado a lado `v1` vs `v2`, rubrica e auditoria).

---

## Arquitetura do Pipeline

```text
+-------------------------------------------------------------------------+
|                        INTERFACES DE USUÁRIO                            |
|   CLI (cli.py / src/cli.py - rich)  |  Web (src/web_app.py - Streamlit) |
+-----------------------------------+-------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                     DADOS E CONTRATOS PYDANTIC v2                       |
|   data/profiles.json   |   data/tickets.json   |   src/models.py        |
+-----------------------------------+-------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|               ORQUESTRADOR DO PIPELINE (src/pipeline/runner.py)         |
|                                                                         |
|   1. Calcula hash SHA-256 canônico (src/pipeline/cache.py)              |
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
|                    +------------+  +----------------------------------+ |
|                    | CACHE HIT  |  | 2. Motor de Prompts Offline      | |
|                    | (0 calls)  |  |    src/prompts/strategies.py     | |
|                    +-----+------+  |    src/prompts/engine.py         | |
|                          |         |    versions/v1.py | v2.py        | |
|                          |         +----------------+-----------------+ |
|                          |                          |                   |
|                          |                          v                   |
|                          |         +----------------------------------+ |
|                          |         | 3. Cliente Desacoplado Gemini    | |
|                          |         |    src/services/llm_client.py    | |
|                          |         |    (google-genai SDK)            | |
|                          |         +----------------+-----------------+ |
|                          |                          |                   |
|                          |                          v                   |
|                          |         +----------------------------------+ |
|                          |         | 4. Persistência Atômica em JSON  | |
|                          |         |    src/services/storage.py       | |
|                          |         |    -> data/history.json          | |
|                          |         +----------------+-----------------+ |
|                          |                          |                   |
|                          +------------+-------------+                   |
|                                       |                                 |
|                                       v                                 |
|                      +--------------------------------+                 |
|                      | 5. Rubrica de Avaliação (1-5)  |                 |
|                      |    src/pipeline/evaluator.py   |                 |
|                      |    (clarity, fit, actionab.)   |                 |
|                      +--------------------------------+                 |
+-------------------------------------------------------------------------+
```

---

## Estrutura do Repositório

```text
.
├── cli.py                                    # Atalho na raiz delegando para src.cli:main()
├── requirements.txt                          # 6 dependências mínimas compatíveis
├── .env.example                              # Variáveis de ambiente sem credenciais reais
├── .gitignore                                # Ignora .venv/, caches, .env e data/history.json
├── README.md                                 # Documentação principal de setup e uso
├── PROMPT_ENGINEERING_NOTES.md               # Análise técnica de prompts (v1 vs v2) e rubrica
├── data/
│   ├── profiles.json                         # 4 perfis contrastantes cobrindo todas as categorias
│   └── tickets.json                          # 4 tickets realistas de suporte técnico
├── samples/                                  # Exemplos JSON reais versionados e avaliados
│   ├── comparison_v1_vs_v2_diagnosis.json    # Comparação controlada v1 vs v2 com rubrica
│   ├── sample_resolution_advanced.json       # Passo a passo v2 para perfil avançado (TKT-103)
│   ├── sample_clarification_layman.json      # Perguntas priorizadas v2 por telefone (TKT-104)
│   └── sample_flow_summary_ascii.json        # Fluxograma ASCII preservado v2 (TKT-101)
├── scripts/
│   └── generate_samples.py                   # Script reproduzível com retry para gerar samples/
├── src/
│   ├── __init__.py
│   ├── models.py                             # Contratos Pydantic v2 (CustomerProfile, ExecutionRecord...)
│   ├── cli.py                                # Interface CLI (modo interativo + flags argparse)
│   ├── web_app.py                            # Interface Web Streamlit (3 abas)
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── engine.py                         # Motor dinâmico build_prompt(...) + regra anti-CoT
│   │   ├── strategies.py                     # 4 funções de estratégia + dispatcher
│   │   └── versions/
│   │       ├── __init__.py                   # Registro de versões (v1, v2)
│   │       ├── v1.py                         # Baseline minimalista
│   │       └── v2.py                         # Versão otimizada com adaptação por perfil/canal
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm_client.py                     # Interface BaseLLMClient + GeminiClient (google-genai)
│   │   └── storage.py                        # JsonHistoryStorage com escrita atômica
│   └── pipeline/
│       ├── __init__.py
│       ├── cache.py                          # generate_cache_key (SHA-256) e find_cached_execution
│       ├── evaluator.py                      # build_rubric_evaluation e evaluate_execution
│       └── runner.py                         # Orquestrador execute_pipeline e PipelineExecutionResult
└── tests/
    ├── test_stage1_setup.py                  # Checkpoint Etapa 1 (setup, fixtures e rubrica)
    ├── test_stage2_prompts.py                # Checkpoint Etapa 2 (motor offline, v1 vs v2)
    ├── test_stage3_pipeline.py               # Checkpoint Etapa 3 (cache, storage, runner e mocks)
    ├── test_stage4_interfaces.py             # Checkpoint Etapa 4 (CLI e helpers Streamlit offline)
    └── test_stage5_deliverables.py           # Checkpoint Etapa 5 (samples/, docs e segurança)
```

---

## Configuração e Execução Passo a Passo

### 1. Pré-requisitos
- **Python 3.10+** (testado em Windows PowerShell e compatível com Linux/macOS).
- Chave válida da **Google Gemini API** (gratuita via Google AI Studio).

### 2. Criar Ambiente Virtual e Instalar Dependências

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

### 3. Configurar Variáveis de Ambiente (`.env`)

Copie o arquivo modelo `.env.example` para `.env` e preencha sua chave real (o arquivo `.env` está protegido no `.gitignore`):

```powershell
Copy-Item .env.example .env
```

Conteúdo esperado do `.env`:
```dotenv
GEMINI_API_KEY=sua_chave_real_aqui
LLM_PROVIDER=google
LLM_MODEL=gemini-2.5-flash
LLM_TEMPERATURE=0.2
```

> **Observação:** Você pode alternar `LLM_MODEL` para qualquer modelo compatível do Google Gemini (ex.: `gemini-2.5-flash` ou `gemini-3.5-flash-lite`).

---

## Como Executar

### Interface CLI (Obrigatória)

A CLI funciona tanto via `python cli.py` quanto via `python -m src.cli`.

**1. Modo Interativo Guiado** (exibe tabelas numeradas com perfis, tickets, tipos e versões):
```powershell
python cli.py
```

**2. Modo Direto via Flags** (exibindo o prompt montado antes da resposta):
```powershell
python cli.py --customer cust_001 --ticket-id TKT-102 --type diagnosis --version v2 --show-prompt
```

**3. Modo Direto com Ticket Livre + Avaliação pela Rubrica:**
```powershell
python cli.py `
  --customer cust_003 `
  --ticket-text "Endpoint /v1/orders retorna HTTP 502 intermitente após deploy." `
  --type resolution `
  --version v2 `
  --clarity 5 --clarity-justification "Passo a passo claro e bem estruturado." `
  --customer-fit 5 --customer-fit-justification "Linguagem técnica adequada ao perfil avançado." `
  --actionability 5 --actionability-justification "Ações verificáveis com critério de escalonamento."
```

**4. Demonstração de Cache Hit e `force-refresh`:**
- Execute o mesmo comando duas vezes seguidas: a 1ª exibe `NOVA EXECUÇÃO API` e grava em `data/history.json`; a 2ª exibe `CACHE HIT` sem chamar a API.
- Passe `--force-refresh` para ignorar o cache e forçar nova geração.

### Interface Web Interativa (Streamlit)

```powershell
streamlit run src/web_app.py
```

A aplicação web abre com 3 abas:
1. **Gerar Atendimento** — Seleção de perfil (com badges de nível, estilo e canal), ticket pré-cadastrado ou livre, tipo, versão, `force_refresh`, inspeção do prompt enviado, renderização monoespaçada automática para `flow_summary` e formulário integrado de rubrica.
2. **Comparação Controlada (`v1` vs `v2`)** — Executa e exibe `v1` e `v2` lado a lado sob exatamente as mesmas configurações de provedor, modelo e temperatura, com formulários individuais de rubrica.
3. **Histórico e Auditoria (`data/history.json`)** — Métricas agregadas, filtros por cliente/tipo/versão/avaliadas, inspeção detalhada e exportação JSON.

### Regenerar Amostras Versionáveis (`/samples/`)

```powershell
# Reutiliza cache dos arquivos em samples/ (rápido e sem custo extra)
python scripts/generate_samples.py

# Força novas chamadas à API Gemini (com retry exponencial em caso de 503)
python scripts/generate_samples.py --force-refresh
```

---

## Testes Automatizados (100% Offline)

Toda a suíte de testes roda **100% offline** (sem gastar créditos de API e sem poluir `data/history.json`), usando dublês determinísticos (`FakeLLMClient`) e diretórios temporários (`tmp_path`):

```powershell
# Executa todos os 25 testes (Etapas 1 a 5)
pytest -v
```

Cada checkpoint também pode ser executado individualmente com resumo visual em `rich`:
```powershell
python tests/test_stage1_setup.py
python tests/test_stage2_prompts.py
python tests/test_stage3_pipeline.py
python tests/test_stage4_interfaces.py
python tests/test_stage5_deliverables.py
```

---

## Decisões Técnicas, Trade-offs e Limitações Conhecidas

- **Desacoplamento de Provedor (`BaseLLMClient`):** A interface abstrata em `src/services/llm_client.py` isola o SDK `google-genai`, permitindo testes determinísticos offline e troca simples para OpenAI/Anthropic sem alterar o pipeline.
- **Cache Determinístico Canônico:** `generate_cache_key` normaliza espaços do ticket e arredonda `temperature` (`round(float(temperature), 4)`) antes de gerar o SHA-256 sobre JSON com `sort_keys=True`, evitando falsos *cache misses* por ruído de ponto flutuante.
- **Preservação Exata de Fluxogramas ASCII:** `ExecutionRecord` intencionalmente **não** aplica `str_strip_whitespace=True` (ao contrário de `CustomerProfile`), preservando espaços iniciais de indentação em `flow_summary`. Na CLI usa-se `rich.text.Text` para que colchetes `[SIM]`/`[NÃO]` não sejam interpretados como markup, e no Streamlit usa-se `st.code(..., language="text")`.
- **Escrita Segura no Histórico JSON:** `JsonHistoryStorage` grava primeiro em arquivo temporário `.tmp` e substitui atomicamente `data/history.json`, evitando corrupção em interrupções de escrita.
- **Limitações Conhecidas:**
  - O histórico e o cache utilizam armazenamento em arquivo JSON local — ideal para portabilidade e inspeção no desafio, porém não indicado para alta concorrência multiusuário (onde um banco como SQLite/PostgreSQL ou Redis seria recomendado).
  - Em modelos muito leves (`gemini-3.5-flash-lite`), observou-se eco parcial das instruções descritivas nos títulos das seções do `diagnosis` na `v2` — achado documentado com transparência em [`PROMPT_ENGINEERING_NOTES.md`](PROMPT_ENGINEERING_NOTES.md) junto à proposta de refinamento para uma futura `v3`.
