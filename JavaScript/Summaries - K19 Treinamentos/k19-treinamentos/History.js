// HISTORY

// 1. Avançar e retroceder no histórico


window.history.forward()
window.history.back()


// 2. window.history.go(-3)


// 3. window.history.length


// 4. Alterando o histórico

state = {info: 'info'}
window.history.pushState(state, title, url)  
window.history.replaceState(state, title, url)


// 5. 