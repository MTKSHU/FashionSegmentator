
$(document).ready(function(){
    $("#send_data").on('click', function(e){
        e.preventDefault();
		e.stopPropagation();
        send_data();
        return true;
    });
});




function send_data(){
    console.log("Click");

    var formData = new FormData();
    // Main magic with files here
    formData.append('pic', $('input[type=file]')[0].files[0]); 
    formData.append('name', $("#name").val());
    formData.append('zip_result', $("#down_archive").prop('checked'));
    $.ajax({
        url: "http://0.0.0.0:8000/images/",
        type:"POST",
        data: formData,
        contentType: "multipart/form-data",
        processData: false,
        success: function(data){
            console.log(data);
            $('#demo-title').html("success");
            return true;
        },
        error: function (XMLHttpRequest, textStatus, errorThrown){
            console.log(XMLHttpRequest)
            $('#demo-title').html("fail");
            return false;

        },
        complete: function( data ,  textStatus ){
            console.log(textStatus)
        }
    });

/*
    $.ajax({
        url: "0.0.0.0:8000/images/",
        type:"POST",
        dataType: "text",
        data: {"name":"ciao"},
        contentType: "application/json",
        success: function(data){
            console.log(data);
            $('#demo-title').html("success");
            return false;
        },
        error: function (XMLHttpRequest, textStatus, errorThrown){
            console.log(XMLHttpRequest)
            $('#demo-title').html("fail");
            return false;

        }
    });*/
}