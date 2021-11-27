// DOM - DOCUMENT OBJECT MODEL


// PARTE 1 - RECUPERANDO ELEMENTOS E ATRIBUTOS.

// 1. "document.getElementById()"
// (Return element by its unique id value)

var titulo = document.getElementById('titulo')
var form = document.getElementById('formulario')

for (i=0; i<form.attributes.length; i++) {
	console.log(form.attributes[i]); console.log(form.attributes[i].name); console.log(form.attributes[i].value);
};


// 2. "document.getElementsByName()"

var inputs = document.getElementsByName('categoria')
console.log(inputs[0])


// 3. "document.getElementsByTagName()"

labels = document.getElementsByTagName('label')
console.log(labels[0].value)
console.log(labels[0].innerHTML)


// 4. "document.getElementsByClassName()"

titulo_lista = document.getElementsByClassName('centralizado')
console.log(titulo_lista[0].innerHTML)
labels = document.getElementsByClassName('label')


// 5. "document.querySelector()"
// (Return first css selector found)
titulo = document.querySelector('.centralizado')  // para seletores de classe
form = document.querySelector('#formulario')  // para seletores de ids
body = document.querySelector('body') // Qualquer nome de tag
console.log(titulo)
console.log(form)


// 6. "document.querySelectorAll()"
// (Return array of all css selectors found)

labels = document.querySelectorAll('.label') //Para seletores de classe.
console.log(labels[0])


// 7. "element.getAttribute()"

label_for = labels[0].getAttribute('for')
console.log(label_for)


// 8. "element.attributes"

var att = Array()
var values = Array()

for (i=0; i<form.attributes.length; i++) {

	att.push(form.attributes[i].name);
	values.push(form.attributes[i].value);

};
console.log(att, values);


// 9. "element.innerHTML"

var titulo_value = document.getElementById('titulo').innerHTML
var labels_values = Array()

for (i=0; i<labels.length; i++) {
	labels_values.push(labels[i].innerHTML);
};
console.log('HTML interno Título:', titulo_value)
console.log('HTML interno labels:', labels_values)


// PARTE 2 - ALTERANDO ELEMENTOS E ATRIBUTOS

// 1. "element.setAttribute('att_name', 'att_value')"

elem_id = 'titulo'
att_name='class'
att_value='justificado'

var titulo = document.getElementById(elem_id)
titulo.setAttribute(att_name, att_value)   // Associando valor a propriedade classe

var titulo_class = titulo.attributes['class'].value  // Retornando valor da propriedade classe
console.log(titulo_class)


// 2. "element.style"
// (Propriedades com nome composto seguem o padrão 'Camel Case'...)
// (... só a primeira palavra com letra minúscula.)

titulo.style.bold = true
titulo.style.color = 'red'
titulo.style.borderColor = 'black'


// 3. "element.innerHTML"

var novo_titulo = 'Novo Título'
document.getElementsByTagName('h1')[0].innerHTML = novo_titulo


// PARTE 3 - REMOVENDO ELEMENTOS

// 1. "element.remove()"

console.log(labels[3])
labels[3].remove()


// 2. "element.parentNode.removeChild(element)"

form = inputs[3].parentNode
form.removeChild(inputs[3])


// PARTE 4 - ADICIONANDO ELEMENTOS

// 1. "document.creatElement(tagName)" & "element.appendChild(element)"

var new_label = document.createElement('label')
new_label.innerHTML = 'Ciência: '
new_label.setAttribute('for', 'ciencia')
new_label.setAttribute('class', 'label')

var new_input = document.createElement('input')
new_input.setAttribute('id', 'ciencia')
new_input.setAttribute('type', 'checkbox')
new_input.setAttribute('name', 'categoria')

form = document.getElementById('formulario')

form.appendChild(new_label)
form.appendChild(new_input)

// 2. "element.insertBefore()"

subtitle = document.createElement('h2')

subtitle.innerHTML = 'Subtitle'
subtitle.setAttribute('id', 'subtitle')
subtitle.setAttribute('class', 'centralizado')

body = document.querySelector('body')
body.insertBefore(subtitle, form)

// END

// EXTRA:

// console.log(document.querySelector('html')) // Return the final html document.

// body_children = body.children

