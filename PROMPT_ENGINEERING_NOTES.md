# Notas de Engenharia de Prompt (`PROMPT_ENGINEERING_NOTES.md`)

Este documento detalha as decisões arquiteturais de engenharia de prompt, as técnicas aplicadas no motor dinâmico, a evolução controlada entre as versões `v1` (baseline) e `v2` (otimizada), a análise comparativa real com a rubrica obrigatória e as limitações observadas em execução real com a API Gemini.

---

## 1. Arquitetura do Motor de Prompts (`src/prompts/engine.py`)

O motor de montagem `build_prompt(...)` em `src/prompts/engine.py` separa **estrutura transversal** (orquestração, segurança e metadados) de **conteúdo versionado** (`src/prompts/versions/v1.py` e `v2.py`) e de **estratégias por artefato** (`src/prompts/strategies.py`).

Cada prompt montado dinamicamente é composto por até **6 blocos modulares**:

1. **`[PROMPT_METADATA]` (Versionamento e Rastreabilidade):**
   Declara explicitamente `prompt_version` (`v1` ou `v2`), `content_type` (`diagnosis`, `resolution`, `clarification` ou `flow_summary`) e a `description` da versão executada. Garante auditabilidade na inspeção via `--show-prompt` (CLI) e no `st.expander` (Streamlit).
2. **`[PERSONA E LIMITES]` (Persona Prompting):**
   Define papel, responsabilidade e fronteiras operacionais do assistente, variando de uma persona genérica na `v1` para um Engenheiro de Suporte Técnico Sênior com limites anti-alucinação explícitos na `v2`.
3. **`[CONTEXTO DO CLIENTE E DO TICKET]` (Context Setting):**
   Injeta todos os atributos validados do `CustomerProfile` (`customer_id`, `name`, `technical_level`, `communication_style`, `channel`) e encapsula o relato do incidente dentro das tags delimitadoras `<ticket>...</ticket>`.
4. **`[ADAPTAÇÃO AO PERFIL E CANAL]` (Calibração Dinâmica Ortogonal):**
   Bloco condicional renderizado automaticamente quando a versão define `build_adaptation_guidelines(profile)`. Presente exclusivamente na `v2` (ausente na `v1`), traduz os três eixos do perfil em regras operacionais concretas.
5. **`[TAREFA E DECOMPOSIÇÃO]` (Task Decomposition):**
   Recebe o `task_instruction` específico da estratégia escolhida (`diagnosis`, `resolution`, `clarification` ou `flow_summary`), orientando exatamente o objetivo cognitivo da resposta.
6. **`[FORMATO DE SAÍDA E RESTRIÇÕES]` (Output Formatting + Regra Anti-CoT):**
   Combina o contrato de saída (`output_format`) da estratégia com as restrições gerais da versão e com a regra transversal `NO_RAW_COT_RULE`, proibindo exposição de raciocínio interno bruto.

---

## 2. Técnicas de Engenharia de Prompt Aplicadas

| Técnica | Implementação no Projeto | Problema Mitigado |
| :--- | :--- | :--- |
| **Persona Prompting com Limites** | `PERSONA` em `v2.py` define papel de Engenheiro de Suporte Sênior + 3 limites estritos (não inventar logs/métricas, declarar lacunas, não prometer prazos fora do escopo). | Alucinação de causas raiz, invenção de evidências ausentes e promessas indevidas de SLA. |
| **Context Setting com Delimitação Anti-Injeção** | Relato do usuário isolado entre `<ticket>...</ticket>` + regra em `v2.py` determinando que todo texto interno é dado, nunca instrução. | *Prompt injection* via tickets maliciosos e ambiguidade entre relato do cliente e diretrizes do sistema. |
| **Task Decomposition** | 4 estratégias independentes em `src/prompts/strategies.py` (`build_diagnosis_prompt`, `build_resolution_prompt`, `build_clarification_prompt`, `build_flow_summary_prompt`). | Respostas monolíticas e confusas que misturam diagnóstico, perguntas e instruções sem foco. |
| **Calibração Dinâmica Ortogonal** | Composição independente de `TECHNICAL_LEVEL_RULES` (`leigo` / `intermediário` / `avançado`) × `CHANNEL_RULES` (`chat` / `e-mail` / `telefone`) × `COMMUNICATION_STYLE_RULES` (`direto` / `detalhado` / `didático`). | Explosão combinatória de templates (evita manter $3 \times 3 \times 3 = 27$ prompts duplicados, mantendo apenas 9 regras ortogonais). |
| **Output Formatting Estrito** | Contratos explícitos por tipo: 4 seções fixas no `diagnosis`, passos com *ação + sinal de sucesso + escalonamento N2* no `resolution`, 3–5 perguntas priorizadas com justificativa no `clarification` e fluxograma ASCII `+ - \| > v` com `[SIM]`/`[NÃO]` no `flow_summary`. | Inconsistência estrutural entre execuções e dificuldade de verificação objetiva. |
| **Supressão de Chain-of-Thought Bruto** | Constante `NO_RAW_COT_RULE` aplicada em `src/prompts/engine.py` proibindo monólogo interno, rascunhos mentais e tags `<thinking>`. | Atendimento ao requisito explícito do edital: priorizar saída estruturada e verificável em vez de raciocínio interno exposto. |

---

## 3. Evolução Controlada: `v1` (Baseline) vs `v2` (Otimizada)

Para permitir uma avaliação experimental justa, `v1` e `v2` compartilham exatamente o mesmo motor (`engine.py`), o mesmo perfil, o mesmo ticket, o mesmo provedor/modelo (`google/gemini-3.5-flash-lite`) e a mesma temperatura (`0.2`).

### 3.1 Diferenças Estruturais Gerais

| Dimensão | `v1` (Baseline — `src/prompts/versions/v1.py`) | `v2` (Otimizada — `src/prompts/versions/v2.py`) |
| :--- | :--- | :--- |
| **Persona** | Genérica ("assistente de suporte técnico"). | Especialista sênior com 3 limites explícitos de veracidade e escopo. |
| **Adaptação por Perfil** | Apenas lista atributos no contexto; não instrui *como* adaptar vocabulário. | Regras explícitas para `leigo` (zero jargão sem tradução), `intermediário` e `avançado`. |
| **Adaptação por Canal** | Ausente (mesmo texto para chat, e-mail e telefone). | Formatação dedicada: blocos curtos em `chat`, estrutura formal em `e-mail`, roteiro falado natural em `telefone`. |
| **Adaptação por Estilo** | Ausente. | Diretrizes operacionais para `direto`, `detalhado` e `didático`. |
| **Proteção contra Injeção** | Apenas tags `<ticket>` do motor. | Tags `<ticket>` + cláusula explícita tratando conteúdo estritamente como relato. |

### 3.2 Contratos das 4 Estratégias (`v1` vs `v2`)

| Estratégia (`ContentType`) | Contrato na `v1` | Contrato na `v2` |
| :--- | :--- | :--- |
| **`diagnosis`** | Texto estruturado com as 4 seções, sem ordenação de hipóteses nem nível de confiança. | Exige 4 seções ordenadas (`## Hipóteses` priorizadas 2–4 itens, `## Evidências Disponíveis` apenas fatos do ticket, `## Dados Faltantes`, `## Conclusão Provável` com nível de confiança e primeira validação). |
| **`resolution`** | Lista numerada simples de passos recomendados. | Passo a passo numerado calibrado ao perfil/canal com **ação concreta + sinal esperado de sucesso** + bloco final `"Se o problema persistir"` para escalonamento N2. |
| **`clarification`** | Lista livre de perguntas ao cliente, sem limite de quantidade ou priorização. | Lista priorizada de **3 a 5 perguntas essenciais**, proibindo repetir dados já informados e exigindo justificativa curta (`Por que destrava o diagnóstico`). |
| **`flow_summary`** | Fluxograma simples ou descrição textual livre. | Fluxograma ASCII limpo dentro de bloco `text` usando apenas `+`, `-`, `|`, `>`, `v`, desvios `[SIM]`/`[NÃO]`, largura recomendada de até 78 colunas e caminho até resolução ou escalonamento N2. |

---

## 4. Avaliação Comparativa com a Rubrica (Evidência Real)

A comparação controlada obrigatória foi executada e persistida em [`samples/comparison_v1_vs_v2_diagnosis.json`](samples/comparison_v1_vs_v2_diagnosis.json) sob condições rigorosamente idênticas:

- **Cliente:** `cust_001` — Mariana Silva (`technical_level: leigo`, `communication_style: didático`, `channel: chat`)
- **Ticket:** `TKT-102` — *Erro 504 Gateway Timeout no webhook de confirmação de pagamentos*
- **Tipo de Conteúdo:** `diagnosis`
- **Modelo / Temperatura:** `google/gemini-3.5-flash-lite` | `temperature = 0.2`

### 4.1 Tabela Comparativa da Rubrica (Escala 1 a 5)

| Critério | Nota `v1` | Justificativa `v1` | Nota `v2` | Justificativa `v2` | Delta |
| :--- | :---: | :--- | :---: | :--- | :---: |
| **Clareza (`clarity`)** | **3 / 5** | Estrutura as 4 seções solicitadas, porém mantém tom genérico e não prioriza explicitamente hipóteses nem indica nível de confiança. | **4 / 5** | Separa rigorosamente hipóteses priorizadas, evidências do relato, dados faltantes e conclusão com nível de confiança explícito (alto); descontou-se 1 ponto porque o modelo ecoou parte da instrução descritiva nos títulos das seções (achado documentado para futura `v3`). | **+1** |
| **Adequação ao Cliente (`customer_fit`)** | **2 / 5** | Usa jargões técnicos sem tradução (`webhook`, `payload síncrono`, `HTTP 504`) e blocos extensos inadequados para cliente leigo em atendimento por chat. | **5 / 5** | Traduz termos técnicos para linguagem acessível e didática (`"endereço digital"`, `"tempo esgotado"`, `"mais de meio minuto"`), em blocos curtos e escaneáveis calibrados ao perfil leigo no canal chat. | **+3** |
| **Acionabilidade (`actionability`)** | **3 / 5** | Aponta o gargalo geral de tempo de resposta, mas deixa a próxima validação pouco objetiva para o atendente conduzir com o cliente. | **5 / 5** | Explicita o nível de confiança (`alto`), lista lacunas objetivas e define a primeira validação recomendada sem exigir conhecimento técnico avançado. | **+2** |
| **Total** | **8 / 15** | Baseline funcional, porém desalinhado ao perfil `leigo/chat`. | **14 / 15** | Ganho expressivo de **+6 pontos (+75%)** sob exatamente o mesmo modelo e temperatura. | **+6** |

### 4.2 Evidências Complementares nas Outras 3 Estratégias (`v2`)

Além da comparação controlada, cada estratégia adicional foi validada em cenários contrastantes na pasta `samples/`:

- **[`samples/sample_resolution_advanced.json`](samples/sample_resolution_advanced.json)** (`cust_003` — `avançado/direto/chat`, `TKT-103`, notas **5/5/5**): entregou 3 passos técnicos precisos (`Authorization: Bearer <token>`, escopos `inventory:write`/`inventory:read`, teste isolado via Postman/cURL) e protocolo explícito de coleta de evidências mascarando tokens antes de escalar para N2.
- **[`samples/sample_clarification_layman.json`](samples/sample_clarification_layman.json)** (`cust_004` — `leigo/direto/telefone`, `TKT-104`, notas **5/5/5**): produziu roteiro falado natural direcionado nominalmente à cliente (`"Fernanda, esses terminais de caixa estão ligados por cabo..."`), respeitando o teto de 3 a 5 perguntas e explicando por que cada resposta destrava o diagnóstico.
- **[`samples/sample_flow_summary_ascii.json`](samples/sample_flow_summary_ascii.json)** (`cust_002` — `intermediário/detalhado/e-mail`, `TKT-101`, notas **5/4/5**): gerou fluxograma ASCII limpo dentro de bloco `text`, com caixas `+ - | v` e desvios `[SIM]`/`[NÃO]` levando até orientação paliativa ou escalonamento N2, com indentação preservada integralmente no JSON.

---

## 5. Limitações Conhecidas, Trade-offs e Próxima Iteração (`v3`)

1. **Eco de Instruções nos Cabeçalhos em Modelos Leves (`gemini-3.5-flash-lite`):**
   Na saída `v2` de `comparison_v1_vs_v2_diagnosis.json`, o modelo reproduziu literalmente o texto instrutivo ao lado do título (ex.: `1. ## Hipóteses — Liste de 2 a 4 hipóteses...`). Em uma futura versão `v3`, isso pode ser eliminado separando o formato literal do título da instrução explicativa (ou usando *Structured Outputs* via `response_schema` JSON do SDK `google-genai`).
2. **Trade-off entre Saída Textual e JSON Estrito:**
   Optou-se por saída textual estruturada em Markdown/ASCII para priorizar legibilidade imediata em chat, e-mail, roteiro telefônico e fluxogramas ASCII, mantendo a estruturação rígida na camada do `ExecutionRecord`. O campo `output: dict[str, Any] | str` em `src/models.py` já está preparado para suportar JSON estruturado em uma evolução futura sem quebra de contrato.
3. **Autoavaliação Humana vs. LLM-as-a-Judge:**
   Conforme orientado pelo desafio, a rubrica foi aplicada com avaliação crítica humana transparente (inclusive penalizando a própria `v2` em Clareza quando houve eco de instrução). Um avaliador automático adicional poderia complementar — mas não substituir — a revisão humana.
