
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
            $("#res_img").attr('src',pic_path);
            $(".child").remove();
            nbb = data['bounding_box'].length;
            offs = Math.round(255/nbb,0);
            r=0;
            g=0;
            b=0;
            for(var i=0;i<data['bounding_box'].length;i++){
                console.log(data['bounding_box'][i]);
                r+=((i)%3==0)*offs;
                g+=((i)%3==1)*offs;
                b+=((i)%3==2)*offs;
                add_rect('rgb('+r+','+g+','+b+')',data['bounding_box'][i]);
            }         
            return false;
        },
        error: function (XMLHttpRequest, textStatus, errorThrown){
            console.log(XMLHttpRequest)
            return false;

        }
    });
}


var add_rect = function(color, rect) {
    var $container = $("#container");
    $('<div class="child"/>')
    .appendTo($container)
    .css("left", rect['min_x'] + "px")
    .css("top", rect['min_y'] + "px")
    .css("width", (rect['max_x']-rect['min_x'])+"px")
    .css("height", (rect['max_y']-rect['min_y'])+"px")
    .css("border", "5px solid " + color);
};