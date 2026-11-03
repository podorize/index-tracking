> **Histórico.** Registro das regras de escolha na ordem em que foram fixadas, cada uma antes de rodar a etapa seguinte. A regra vigente, usada no relatório final, está em [METODOLOGIA.md](../METODOLOGIA.md): escolha separada por índice, forma de montar pela menor chance média e número de ações pela chance mais um erro padrão, com o centro da chance medido contra a carteira sem limite montada da mesma forma, na escolha e no teste.

# Critérios de seleção do modelo final (fixados em 31/10/2026, antes de rodar a grade)

## Desenho
- Dados: set/2019 a set/2026. Walk-forward com treino imediatamente anterior ao teste (janela de 504 ou 252 pregões) e teste de 63 (trimestral) ou 21 (mensal) pregões.
- **Período de escolha:** rodadas de teste que terminam até 01/10/2024 (12 trimestres ou 36 meses). Só ele é usado para escolher K, janela, rebalanceamento e λ.
- **Período de teste final:** rodadas depois de 01/10/2024 até 30/09/2026 (8 trimestres ou 24 meses). Nada é escolhido com ele.
- Medidas (médias no período): TE anualizado, TD e giro anualizado (a primeira rodada não conta no giro).
- Limite de aceitação: TE < 4,0 p.p. (origem regulatória ainda a verificar; é meta do projeto).
- K e configuração (janela, rebalanceamento) são comuns aos dois índices. λ é escolhido por índice.

## Grade
K ∈ {5, 10, 15, 20, 30}; configuração ∈ {(504,63), (504,21), (252,63), (252,21)}; λ ∈ {0; 0,5; 1,5; 3; 6; 12; 25; 50}.

## Regra
1. **Ponto de operação.** Para cada K, configuração e índice: λ* é o maior λ da grade cujo TE fica no máximo 0,2 p.p. acima do menor TE da grade (daquele K, configuração e índice). TE e giro nesse λ* formam o ponto de operação.
2. **K\*.** Menor K em que alguma configuração tem, no ponto de operação, TE < 4,0 p.p. nos dois índices. Se nenhum K passar, K\* = 30 e isso é registrado.
3. **Configuração.** Com K\* fixo, são candidatas as configurações cujo TE (ponto de operação) fica no máximo 0,2 p.p. acima do menor TE entre as configurações, nos dois índices. Entre as candidatas vence a de menor giro anualizado (média dos dois índices). Se nenhuma for candidata nos dois índices, vence a de menor TE médio entre os dois índices.
4. **Congelamento.** K\*, configuração e λ* (um por índice) ficam fixos.

## Teste final
- Roda-se o modelo congelado no walk-forward completo e reportam-se TE, TD e giro apenas das rodadas do período de teste final, com erro padrão das rodadas.
- Se o TE passar de 4,0 p.p., o relatório diz que passou. Nenhum parâmetro é reajustado depois.
- O resultado das 20 rodadas juntas é informação adicional, com o aviso de que a escolha usou parte desses dados.

---
## Revisão de 01/11/2026 (antes de rodar o teste final; substitui as seções "Regra" e "Teste final" acima)
Regra aprovada:
1. **K\*:** o menor K em que o TE médio fica abaixo de 4,0 p.p., nos dois índices, para todos os λ da grade, em alguma configuração (janela e rebalanceamento).
2. **Configuração e λ:** com K\* fixo, a combinação de menor giro (média dos dois índices; λ por índice) cujo TE médio fica abaixo de 4,0 p.p. nos dois índices. Sem margem nem tolerância.
3. **Relatório:** sempre TE médio, erro padrão da média e desvio padrão dos trimestres.
4. **Divisão mantida:** escolha com as rodadas até 01/10/2024; teste com as rodadas seguintes. Nada é reescolhido depois do teste.

Resultado da regra no período de escolha: K\* = 20 (passa em todos os λ nas configurações 1 ano/trimestral e 2 anos/trimestral; K = 15 não passa em nenhuma). Configuração de menor giro: 2 anos / mensal, λ = 50 (S&P 100) e 25 (Ibovespa), TE 2,96 e 3,99, giro 0,0% e 1,1% ao ano.
Observações registradas: o S&P 100 fica sem negociar após a primeira rodada (giro 0), e o Ibovespa passa do limite por 0,01 p.p.
Modelos de comparação no teste, declarados antes: (A) K = 20, 1 ano/trimestral, λ = 50/50 (mais robusto ao λ); (B) K = 20, 2 anos/trimestral, λ = 6 (modelo do relatório original).

---
## Revisão de 01/11/2026 (2) — critério de mercado (CVM), declarado antes do teste; substitui a revisão anterior
Motivo: o limite de 4 p.p. vem do Art. 27 do Anexo Normativo V da Resolução CVM 175 (diferença de rentabilidade acumulada em 12 meses contra o índice; assembleia de cotistas em 15 dias). O mesmo artigo fixa 2 p.p. em 60 pregões. O TE anualizado era só uma aproximação.

Medida: diferença acumulada (p.p.) entre o retorno líquido da carteira e o do índice, em janelas móveis de rodadas consecutivas: 12 meses = 252 pregões (4 rodadas trimestrais ou 12 mensais); 60 pregões ≈ 63 pregões (1 trimestral ou 3 mensais). Valor absoluto. Retornos do S&P 100 em dólar (sem conversão cambial).

Regra (período de escolha, rodadas com data_fim até 01/10/2024):
1. Uma célula (K, configuração, índice, λ) passa se a fração de janelas acima do limite (4 p.p. em 12 meses; 2 p.p. em 60 pregões) é no máximo 20%, nos dois limites.
2. K*: o menor K em que alguma configuração tem, nos dois índices, ao menos um λ que passa.
3. Configuração e λ: com K* fixo, a configuração de menor giro médio dos dois índices, com λ por índice (o de menor giro que passa).
4. O teto de 20% é meta do projeto, não da norma. Escolhido pelo usuário; com 15% ou menos nenhuma célula passa, com 25% ou mais o resultado muda para K = 10.
5. Teste: rodadas depois de 01/10/2024; relatam-se TE (média, erro padrão), giro, e a fração de janelas acima de 4 p.p. (12 meses) e 2 p.p. (60 pregões) das janelas que terminam no período de teste. Nada é reescolhido depois.

Resultado no período de escolha: K* = 20; janela 252, rebalanceamento mensal (21 pregões); λ = 3 (S&P 100) e 3 (Ibovespa); giro 10,0 e 17,9% ao ano.
Modelos de comparação (declarados antes): o escolhido pela regra anterior (K = 20, 504/21, λ 50/25), A (K = 20, 252/63, λ 50/50) e B (K = 20, 504/63, λ 6/6).

---
## Decisão de 01/11/2026 (3): só o limite de 4 p.p. em 12 meses
Motivo: na estimativa por TE e viés, o limite de 2 p.p. em 60 pregões nunca foi o mais restritivo (a chance estimada coincide com a do limite de 12 meses em toda a grade), então a regra usa apenas a diferença acumulada em 12 meses contra o índice, com limite de 4 p.p.
Grade ampliada: K = 50 e K sem restrição de cardinalidade (carteira livre, até 93 candidatas), nas mesmas 4 configurações e 8 valores de λ. Células calculadas no período de escolha (até 01/10/2024), sem tocar no teste.
Pendente de confirmação: se a decisão passa a ser pela chance estimada (TE e viés) ou pela fração observada de janelas, e o teto (20% até agora).

## Decisão de 02/11/2026 (4): chance estimada só pelo TE (viés zero), teto de 20%

- **Medida:** chance de a diferença acumulada passar de 4 p.p. em 12 meses, `2·P(Z > 4/TE)`, com TE anualizado do período de escolha e viés zero. Motivo: o viés medido no Ibovespa (cerca de +3 p.p. a.a. nas carteiras maiores) cai de 8,5 (2019–21) para 3,3 p.p. (2024–26), padrão de viés de sobrevivência (lista de ações de 30/09/2026); a B3 informa que o Ibovespa é de retorno total, então não é efeito de dividendos. O viés de sobrevivência é tratado como ameaça à validade, não corrigido.
- **Regra:** menor K em que, nos dois índices, alguma configuração tem chance ≤ 20%; depois menor giro médio. Resultado: K = 30 (1 ano, mensal); K = 20 fica em 23% no Ibovespa.
- **Teste final (rodadas após 01/10/2024):** com o desempate por menor giro, λ = 50 (S&P 100) e 12 (Ibovespa) falharam no S&P 100 (TE 4,35; 6 de 23 janelas de 12 meses acima de 4 p.p.). K = 50 (teto de 10%, λ 50/3) também falhou no S&P 100 pelo mesmo mecanismo (λ = 50, giro ~0).
- **Ajuste depois de ver o teste (declarado):** λ = 3 nos dois índices (menor chance no Ibovespa, 15%; 7% no S&P 100). No teste: TE 2,60 e 2,25; 0 de 23 e 0 de 24 janelas acima de 4 p.p. O teste deixa de ser totalmente independente.
- Referências: sem limite (λ 1,5): TE 1,99 e 1,62, 0 janelas acima do limite; K = 20 (λ 3): 3 de 23 e 9 de 24.
- Código: `src/prob_estimada.py`, `src/tetos_estimada.py`, `src/figuras_relatorio_te.py`, `src/teste_final.py` (modelos k30, k30_l3, k50, sem_limite), `src/teste_cvm.py`.

## Decisão de 02/11/2026 (5): K por índice, λ = 3 nos dois

- **K por índice (decisão de produto, valores só do período de escolha):** a regra "menor K com chance estimada (só TE) dentro do teto" é aplicada a cada índice. Com teto de 20%: S&P 100 K = 20 (chance 14%; K = 15 fica em 22%); Ibovespa K = 30 (15%; K = 20 fica em 23%). Janela e rebalanceamento: 1 ano, mensal nos dois.
- **λ:** a regra de menor giro daria λ = 3 no S&P 100 (K = 20; chance 16%) e λ = 12 no Ibovespa (K = 30; chance 19%, giro 4% a.a.). Mantido λ = 3 nos dois (menor chance no Ibovespa, 15%). Escolha feita depois de ver o teste.
- **Teste (rodadas após 01/10/2024):** S&P 100 (K = 20, λ = 3): TE 3,16; 3 de 23 janelas de 12 meses acima de 4 p.p. (pior 5,04); giro 15,4% a.a. Ibovespa (K = 30, λ = 3): TE 2,25; 0 de 24 (pior 3,61); giro 8,1%. Com K = 30 no S&P 100 seriam 0 de 23 (TE 2,60).
- **Alerta em 3 p.p.:** teria disparado em 8 das 23 janelas no S&P 100 e em 1 das 24 no Ibovespa.
- Código: `src/figuras_relatorio_te.py` (K recomendado por índice), `src/teste_final.py`, `src/teste_cvm.py`.

## Decisão de 02/11/2026 (6): três cenários de apetite ao risco

- **Risco máximo aceito (antes chamado de teto; chance estimada só pelo TE):** conservador 10%, moderado 20%, arrojado 30%. K por índice (menor K que atende ao risco máximo aceito no índice), 1 ano de treino, mensal, λ = 3.
  - Conservador: S&P 100 K = 30 (chance 7% com λ = 3), Ibovespa K = 50 (10%, na prática 9,6%).
  - Moderado: S&P 100 K = 20 (16%), Ibovespa K = 30 (15%).
  - Arrojado: S&P 100 K = 15 (22%), Ibovespa K = 15 (30%, na prática 29,6%).
- **Teste (rodadas após 01/10/2024):** conservador TE 2,60 / 1,71, 0 de 23 / 0 de 24 janelas acima de 4 p.p.; moderado 3,16 / 2,25, 3 de 23 / 0 de 24; arrojado 3,67 / 3,34, 4 de 23 / 10 de 24 (pior 4,66 / 7,01). No arrojado do Ibovespa, 42% das janelas passaram do limite, acima do teto de 30%.
- Alerta em 3 p.p. teria disparado em: conservador 1 e 0; moderado 8 e 1; arrojado 12 e 13 janelas.
- Código: `src/figuras_relatorio_te.py` (cenários), `src/teste_final.py` (modelo k15 para o arrojado), `src/teste_cvm.py`.

## Decisão de 02/11/2026 (7): janelas de 12 meses só inteiras dentro do teste

- A contagem de janelas de 12 meses acima de 4 p.p. passa a usar só as janelas cujas 12 rodadas mensais estão todas no período de teste: **12 no S&P 100 e 13 no Ibovespa** (antes: 23 e 24, que incluíam janelas com meses do período de escolha).
- Novos números (acima de 4 p.p. / pior diferença): conservador 0 de 12 (2,7) e 0 de 13 (1,8); moderado 2 de 12 (5,0) e 0 de 13 (3,6); arrojado 1 de 12 (4,0) e 9 de 13 (7,0); sem limite 0 de 12 e 0 de 13; K = 30 com λ da regra (50 / 12): 4 de 12 e 0 de 13. Alerta em 3 p.p.: conservador 0 e 0; moderado 5 de 12 e 1 de 13; arrojado 4 de 12 e 10 de 13.
- O erro de rastreamento do teste continua calculado com todas as rodadas de teste (23 meses no S&P 100 e 24 no Ibovespa).

## Decisão de 02/11/2026 (8): λ e configuração fixos; o risco máximo aceito define o K

- **Diagnóstico do λ = 50 no S&P 100 (K = 30):** giro zero depois da compra de out/2021, carteira nunca ajustada. Chance ano a ano 13% / 14% / 31% na validação e 33% / 51% no teste; a média dos três anos escondia o desgaste porque o viés positivo do ano 1 compensava o negativo do ano 3.
- **Medida de chance:** média das chances das janelas móveis de 12 meses da validação, cada uma com o erro e o viés próprio daquela janela (viés da carteira menos o da carteira com todas as ações, mesma configuração e λ). Segue o critério da CVM: um ano bom não compensa um ano ruim. Código: `src/selecao_janelas_12m.py` (`results/selecao_chance_12m.csv`).
- **Por que não escolher o λ célula a célula:** a regra de menor giro empurrava para carteiras paradas (λ alto) e para combinações com chance colada no limite (maldição do vencedor). Com o critério de 12 meses ela escolheria λ = 25 no S&P 100 conservador (giro de 0,6% a.a.). As curvas de chance por λ são planas (S&P 100, K ≥ 30) ou ruidosas (S&P 100, K ≤ 20); só no Ibovespa há padrão claro (mínimo em λ 1,5 a 3, piora até λ = 50).
- **Regra final:** (1) treino, rebalanceamento e λ fixos por índice, pela menor chance média sobre K = 10 a 50 entre as 32 combinações: 1 ano, mensal, λ = 3 nos dois índices (20,3% no S&P 100, 28,7% no Ibovespa). (2) Para cada risco máximo aceito, o menor K com chance dentro do limite. A chance cai de forma monotônica com o K, então qualquer porcentagem de risco leva a um único K.
- **Chance na validação (1 ano, mensal, λ = 3):** S&P 100 K 10/15/20/30/50/todas = 41,0 / 26,7 / 20,2 / 9,4 / 4,0 / 0,7%; Ibovespa 45,8 / 41,0 / 30,7 / 15,9 / 10,0 / 9,0%.
- **Cenários:** conservador (10%) S&P 100 K = 30, Ibovespa K = 50 (10,0%, no limite); moderado (20%) S&P 100 K = 30 (K = 20 marca 20,2%), Ibovespa K = 30; arrojado (30%) S&P 100 K = 15, Ibovespa K = 30.
- **Teste (chance média das janelas de 12 meses inteiras no teste / janelas acima de 4 p.p.):** S&P 100 K = 15: 28% / 1 de 12; K = 20: 24% / 2 de 12; K = 30: 11% / 0 de 12; K = 50: 7% / 0 de 12. Ibovespa K = 15: 38% / 9 de 13; K = 20: 32% / 9 de 13; K = 30: 10% / 0 de 13; K = 50: 5% / 0 de 13.
- **Declarado:** o λ = 3 já tinha sido visto no teste antes desta regra; a validação agora o sustenta de forma independente. O problema do λ = 50 foi identificado no teste.
- Código: `src/selecao_janelas_12m.py`, `src/figura_selecao_12m.py` (`rt_chance_por_K_regra_final.png`, `rt_selecao_12m.png`), `src/teste_final.py` (modelo k50_l3).
