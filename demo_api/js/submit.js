
$(document).ready(function(){
    $("#send_data").on('click', function(e){
        e.preventDefault();
		e.stopPropagation();
        send_data();
    });

    $('input[type=file]').change(function() { 
        // select the form and submit
        $('#urls').prop("disabled",true);
    });

    $("#clear_btn").on('click', function(e){
        $('#urls').prop("disabled",false);
        $('input[type=file]').prop("disabled",false);
    });

    $('#urls').on("change",function(){
        if($(this).val() == "")
            $('input[type=file]').prop("disabled",false);
        else
        $('input[type=file]').prop("disabled",true);
    })

});




function send_data(){
    console.log("Click");
    var formData = new FormData();
    // Main magic with files here
    var base_url = "http://localhost:8000/";

    if($('#urls').val() == ""){
        formData.append('pic', $('input[type=file]')[0].files[0]); 
        formData.append('name', $('input[type=file]')[0].files[0].name.split('.')[0]);
    }
    else
        formData.append('name', $('#urls').val().split('/').pop().split('.')[0]);
    formData.append('urls', $('#urls').val());
    formData.append('zip_result', false);
    var result = 
        $.ajax({
        url: base_url + "images/",
        type:"POST",
        data: formData,     
        contentType: false,
        processData:false,
        dataType:"json",
        cache:false,
        success: function(data){
            //http://localhost:8000/media/pics/""
            console.log(data);
            var pic_path = base_url + data['pic'].split('/')[5] + '/' + data['pic'].split('/')[6] + '/' +data['pic'].split('/')[7];
            console.log(pic_path);
            $("#res_img").attr('src',pic_path);
            return false;
        },
        error: function (XMLHttpRequest, textStatus, errorThrown){
            console.log(XMLHttpRequest)
            return false;

        }
    });
}