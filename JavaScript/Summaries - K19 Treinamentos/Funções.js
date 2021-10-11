// FUNÇÕES

// 1. Função como método (função dentro de um objeto).

var conta = {
	
	saldo : 0 ,
	deposita : function ( valor ) {
		this.saldo += valor ;
	}

}

conta.deposita(100) ;
console.log( conta.saldo ) ;


// 2. "Apply" (Aplicando função que acesssa propriedades do objeto). 
// Uso: função.Apply(objeto, lista_de_paramêtros)

var deposita = function(valor) { this.saldo += valor; };

deposita.Apply(conta, [100]) // List of arguments to pass to the function.

// Outra forma, porém ambígua.
conta.deposita.Apply(conta, [200])


// 3.Retrieve function arguments' values as array using "arguments" keyword.

print_args = function () {
	
	for (let i=0; i<arguments.length; i++) {
		console.log(arguments[i]);
	};

}; 

print_args(0,1,2,3,4,5);