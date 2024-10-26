$(document).ready(function() {
    $('#submit-button').click(function() {
        const text = $('#input-text').val();
        if (text) {
            $.ajax({
                url: '/api/predict',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({ text: text }),
                success: function(response) {
                    $('#output-div').text('Predicted Sentiment Class: ' + response.predicted_class);
                },
                error: function() {
                    $('#output-div').text('Error in prediction');
                }
            });
        } else {
            $('#output-div').text('Please enter some text');
        }
    });
});
