// ARRAYS

// 1. Criando um Array

var array = [0,1,2,3,4,5,6,7,8,9];


// 2. Adicionando um item ao final do array. (Duas formas)

array[array.length] = 10;
array.push(11);

console.log(array);


// 3. Deletando elementos com o método "delete".
// (Item correspondente ao índice é substituido por "vazio", tamanho do array não muda)

delete array[11];
console.log(array);
console.log(array.length);


// 4. Método "Array.concat()".
// (Concatenando arrays)

var array_2 = [11, 12, 13, 14, 15];

array = array.concat(array_2);
console.log(array);


// 5. Método join()
// (Cria uma string unindo os elementos de um array com um separador)
array_as_string = '[' + array.join(', ') + ']';
console.log(array_as_string);


// 6. Método "pop()"
// (Remove e retorna o último elemento de um array)

last_element = array.pop();
console.log(last_element);


// 7. Método "shift()" - Contrário do método "pop()"
// (Remove e retorna primeiro elemento do array)

first_element = array.shift();
console.log(first_element);


// 8. Método "reverse()"
// (Inverte o array sem retorna-lo)

array.reverse();
console.log(array);


// 9. Método "slice()"
// Copia uma fatia do array)

array_slice = array.slice(3, 8);
console.log(array_slice);


// 10. Método "splice()"
// Argumentos: 
// 1. Índice do primeiro elemento que será substituído.
// 2. Número total de elementos subsequentes que serão substituídos.
// 3. Elementos que se´~ao adicionados nas posições indicadas.

array.splice(2, 3, 'i', 'i', 'i');
console.log(array);


// 11. Método "unshift()"
// (Adiciona elementos na primeira posição de um array)

array.unshift('Primeira posição');
console.log(array);
