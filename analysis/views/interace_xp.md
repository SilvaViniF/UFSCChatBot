# Fluxo de Interação com o Assistente Virtual Acadêmico
```mermaid
%%{init: {'theme': 'neutral', 'themeVariables': { 'primaryColor': '#fff' }}}%%
graph TD
    A[("fa:fa-user Usuário")] --> B[("fa:fa-chrome Abre navegador web")]
    B --> C[["fa:fa-university Acessa portal<br>UFSC Blumenau"]]
    C --> D[("fa:fa-robot Clica no ícone<br>do assistente virtual")]
    D --> E[["fa:fa-comments Interface do Chat<br>(Pergunta digitada)"]]
    E --> F[["fa:fa-server Processamento<br>do sistema RAG"]]
    F --> G[["fa:fa-reply Resposta do assistente<br>Tempo: 2-5s"]]
    G --> H[["fa:fa-clock Registro do tempo<br>de resposta"]]
    H --> I[["fa:fa-eraser Reinicialização<br>do histórico"]]
    I --> J[("fa:fa-check-circle Fim do teste")]

    style A fill:#005CAF,color:white
    style B fill:#005CAF,color:white
    style C fill:#005CAF,color:white
    style D fill:#3F704D,color:white
    style E fill:#3F704D,color:white
    style F fill:#6C3483,color:white
    style G fill:#3F704D,color:white
    style H fill:#B7950B,color:white
    style I fill:#3F704D,color:white
    style J fill:#1D8348,color:white
```
**Legenda de Cores**:
- 🔵 Azul: Passos institucionais
- 🟢 Verde: Interações com o assistente
- 🟣 Roxo: Processamento técnico
- 🟡 Amarelo: Registro de métricas