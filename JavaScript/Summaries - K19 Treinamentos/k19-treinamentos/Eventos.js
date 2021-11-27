// EVENTOS


// 1. Adicionando expressão à evento em elemento, direto no código HTML.

html_text = "<p onclick='console.log('click')'>Isto é um parágrafo</p>"


// 2. element.addEventListener('event', expression)   &   element.removeEventListener('event', expression)

input = document.getElementById('input_id') ;

log_mouseover = function() {console.log('mouse over')})

input.addEventListener('mouseover', log_mouseover) ;
input.removeEventListener('mouseover', log_mouseover) ;


// 3. Associando expressão à evento de elemento, alterando o valor da propriedade do elemento

click_expr = function () {console.log('click')}

input.onclick = click_expr


// 4. Eventos

eventos = ['load', 'resize', 'scroll', 'focus', 'change', 'blur',
		   'select', 'submit', 'reset', ('mousedown', 'mouseup', 'click'),
		   'dblclick', ('mousemove', 'mouseover', 'mouseout'), 
		   ('keydown', 'keypress', 'keyup')


// 5. Exemplos

