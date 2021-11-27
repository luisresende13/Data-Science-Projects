// WEB STORAGE

// 1. Armazenando dados  (localStorage.getItem('key', 'value'))

localStorage.setItem('chave', 'valor')
sessionStorage.setItem('chave', 'valor')

localStorage.chave = 'valor'
sessionStorage.chave = 'valor'


// 2. Recuperando dados  (localStorage.getItem('key', 'value'))

localStorage.getItem('chave')
sessionStorage.getItem('chave')

value = localStorage.chave
value = sessionStorage.chave

chave_0 = localStorage.key(0)
valor_0 = localStorage.getItem(chave_0)


// 3. Excluindo Dados

localStorage.removeItem(chave) // Remove valor por chave.
localStorage.clear() // Limpa armazenamento.
