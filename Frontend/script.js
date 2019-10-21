const formulario = document.getElementById('formulario');
const respuesta = document.getElementById('respuesta');

formulario.addEventListener('submit', function(e){
    e.preventDefault();
    console.log('click precioso');

    let datos = new FormData(formulario)
    console.log(datos);
    let data = []
    data.push(parseFloat(datos.get('CRIM')))
    data.push(parseFloat(datos.get('ZN')))
    data.push(parseFloat(datos.get('INDUS'))) 
    data.push(parseFloat(datos.get('CHAS')))
    data.push(parseFloat(datos.get('NOX')))
    data.push(parseFloat(datos.get('RM')))
    data.push(parseFloat(datos.get('AGE')))
    data.push(parseFloat(datos.get('DIS')))
    data.push(parseFloat(datos.get('RAD')))
    data.push(parseFloat(datos.get('TAX')))
    data.push(parseFloat(datos.get('PTRATIO')))
    data.push(parseFloat(datos.get('B')))
    data.push(parseFloat(datos.get('LSTAT')))

    let envio = JSON.stringify({
        'CRIM': data[0],
        'ZN': data[1],
        'INDUS': data[2],
        'CHAS': data[3],
        'NOX': data[4],
        'RM': data[5],
        'AGE': data[6],
        'DIS': data[7],
        'RAD': data[8],
        'TAX': data[9],
        'PTRATIO': data[10],
        'B': data[11],
        'LSTAT': data[12]
    })

    console.log(data);
    console.log(envio);

    
    fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: envio
        }).then(res => res.json())
            .then(data => {
                console.log(data);
                respuesta.innerHTML = `
                <div class="alert alert-success mt-4" role="alert">
                    $ ${data.precio} Dolares
                </div>
                `
            }).catch(err => {
                console.log('Fetch Error :-S', err);
            });
});