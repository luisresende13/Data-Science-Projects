// OBJETOS

// 1. Criando um Objeto

var objeto = {
	propriedade1: 'valor1',
	propriedade2: 'valor2',
	propriedade3: 'valor3',
};

// 2. Recuperando valores de propriedades

console.log(objeto.propriedade1);
console.log(objeto['propriedade1']);

// 3. Recuperando chaves do objeto

console.log(objeto.key(0))
console.log(objeto[objeto.key(0)])

// 3. Duas maneiras de copiar objetos

var novo_objeto1 = {}
novo_objeto1.__proto__ = objeto

var novo_objeto2 = Object.create(objeto)

console.log(novo_objeto1.propriedade2)
console.log(novo_objeto2.propriedade2)

// 4. Se o objeto pai for modificado o objeto filho também será. O contrário não é verdade.

objeto.propriedade1 = 'novo valor'
console.log(novo_objeto1.propriedade1)

//5. Se o objeto filho for modificado, modificações no objeto pai não alteram mais o objeto filho.

novo_objeto1.propriedade1 = 'novo valor'
objeto.propriedade1 = 'valor1'
// Propriedade do objeto pai alterada e do objeto filho não.
console.log(novo_objeto1.propriedade1)  

// 6. Removendo uma Propriedade com a função delete.

delete objeto.propriedade3;

// 7. Verificando se o objeto possui uma propriedade.

console.log('propriedade3' in objeto)

// END