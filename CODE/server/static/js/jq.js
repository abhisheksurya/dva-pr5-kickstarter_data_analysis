//ref: https://stackoverflow.com/questions/166221/how-can-i-upload-files-asynchronously
$(function(){
$(':button').on('click', function() {
    $.ajax({

        url: '/upload',
        type: 'POST',

        // Form data
        data: new FormData($('form')[0]),

        // Tell jQuery not to process data or worry about content-type
        // You *must* include these options!
        cache: false,
        contentType: false,
        processData: false,

        // Custom XMLHttpRequest
        xhr: function() {
            var myXhr = $.ajaxSettings.xhr();
            if (myXhr.upload) {
                // For handling the progress of the upload
                myXhr.upload.addEventListener('progress', function(e) {
                    if (e.lengthComputable) {
                        $('progress').attr({
                            value: e.loaded,
                            max: e.total,
                        });
                    }
                } , false);
            }
            return myXhr;
        },

        success: function (data) {
            // Contains CSV or JSON data to be fed into D3 for visualization 2
            let vis2 = data.vis2;
            // Contains a base64 encoded image
            let vis3 = data.vis3;
            $("#vis3").attr("src", vis3);
            
        }
    });
});
});
