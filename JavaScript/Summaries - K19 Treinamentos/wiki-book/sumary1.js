



body = document.querySelector('body')
body.style.background.color='Red'

num = 10
float_num = parseFloat(num)
int_num = parseInt(num)

p1 = document.createElement('p')
p1.innerHTML = 'Numero Float: '+String(float_num)+' ---- Numero Interoi: '+String(int_num)

body.appendChild(p1)


// Error Handling

const constante = 3
try { var constante = 10 ;} catch(e) {console.log("Can't divide by zero!");}