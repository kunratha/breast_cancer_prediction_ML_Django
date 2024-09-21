
document.getElementById('feature-form').addEventListener('submit', function (event) {
    event.preventDefault(); // Prevent form from submitting normally
    var form = event.target;

    // Collect form data
    var formData = new FormData(form);

    // Send form data via AJAX
    fetch(form.action, {
        method: 'POST',
        body: formData,
        headers: {
            'X-CSRFToken': formData.get('csrfmiddlewaretoken'),
        }
    })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                document.getElementById('messageContent').innerText = `Error: ${data.error}`;
            } else {
                var result = data.result === "M" ? "Malignant" : "Benign";
                document.getElementById('messageContent').innerText = `The predicted diagnosis is: ${result}`;
            }
            var messageModal = new bootstrap.Modal(document.getElementById('messageModal'));
            messageModal.show();
        })
        .catch(error => console.error('Error:', error));
});
