# Extensão HÍBRIA

## Desenvolvimento

```bash
npm ci
npm run lint
npm run build
```

O build carregável pelo Chrome é criado em `extension/dist`.

## Teste local no Chrome

1. Abra `chrome://extensions`.
2. Ative **Modo do desenvolvedor**.
3. Clique em **Carregar sem compactação**.
4. Selecione a pasta `extension/dist`.

A URL padrão da API é `https://hibria-tcc.duckdns.org`. Para apontar a
extensão a outro servidor durante o desenvolvimento, crie um `.env.local`:

```env
VITE_HIBRIA_API_URL=https://seu-dominio.example
```

Depois execute `npm run build` novamente. Não coloque segredos em variáveis
`VITE_*`: tudo que entra no bundle da extensão pode ser lido pelo usuário.
