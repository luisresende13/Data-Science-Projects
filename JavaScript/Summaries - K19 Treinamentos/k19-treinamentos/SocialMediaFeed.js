

var titulos = Array()
for (var i=0; i<20; i++) {
	titulos.push('Titulo ' + String(i))
}


for (var i=0; i<titulos.length; i++){


	var block = document.createElement('div')

	block_att = ['id', 'class', 'name']
	block_values = ['div_'+String(i), 'block', 'block_'+String(i)]
	for (var j=0; j<block_att.length; j++) { block.setAttribute(bloack_att[j], block_values[j]) }

	let block_style = ['background-color', 'width', 'height', 'left', 'position', 'top']
	let block_style_values = ['red', '300px', '300px', '400px', 'absolute', String(100+400*i)+'px']
	for (var j=0; j<block_style.length; j++) { block.style[block_style[j]] = block_style_values[j] }


	var title = document.createElement('h3')

	title.setAttribute('id', 'title_'+String(i))
	title.setAttribute('class', 'title')
	title.setAttribute('name', 'title_'+String(i))

	title.innerHTML = titulos[i]


	var body = document.querySelector('body')
	block.appendChild(title)
	body.appendChild(block)

}

console.log(body)