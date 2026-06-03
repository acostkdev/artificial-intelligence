const vscode = require("vscode");

const API_URL = "http://127.0.0.1:5000/autocompletar";

function activate(context) {
    let comando = vscode.commands.registerCommand(
        "autocompletado-rnn-c.autocompletar",
        async function () {
            let editor = vscode.window.activeTextEditor;
            if (!editor) {
                vscode.window.showErrorMessage("No hay un editor abierto");
                return;
            }

            let documento = editor.document;
            let cursorPos = editor.selection.active;
            let rango = new vscode.Range(0, 0, cursorPos.line, cursorPos.character);
            let codigo = documento.getText(rango);

            if (!codigo || codigo.trim() === "") {
                vscode.window.showErrorMessage("Escribe algo antes de autocompletar");
                return;
            }

            vscode.window.showInformationMessage(
                "Enviando codigo a la RNN para autocompletar..."
            );

            try {
                let respuesta = await fetch(API_URL, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        codigo: codigo,
                        max_tokens: 200,
                    }),
                });

                if (!respuesta.ok) {
                    vscode.window.showErrorMessage(
                        "Error en el servidor: " + respuesta.status
                    );
                    return;
                }

                let datos = await respuesta.json();
                let completado = datos.completado;

                if (!completado || completado.trim() === "") {
                    vscode.window.showInformationMessage(
                        "La RNN no genero nada nuevo"
                    );
                    return;
                }

                editor.edit(function (edicion) {
                    let posicion = editor.selection.active;
                    edicion.insert(posicion, completado);
                });

                vscode.window.showInformationMessage(
                    "Completado insertado (" + completado.length + " caracteres)"
                );
            } catch (error) {
                vscode.window.showErrorMessage(
                    "No se pudo conectar con la API: " + error.message
                );
            }
        }
    );

    context.subscriptions.push(comando);
}

function deactivate() {}

module.exports = {
    activate,
    deactivate,
};
