
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
    });

    $("#box_result").prop("hidden",true);
    
});




function send_data(){
    console.log("Click");
    var formData = new FormData();
    // Main magic with files here
    var base_url = "http://localhost:8000/";
    $("#waitbar").removeClass('hidden');
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
            $("#box_result").prop("hidden",false);
            var pic_path = base_url + data['pic'].split('/')[5] + '/' + data['pic'].split('/')[6] + '/' +data['pic'].split('/')[7];
            $("#res_img").attr('src',pic_path);
            $("li").remove();
            for(var i=0;i<data['bounding_box'].length;i++){                
                add_label_info(i,data['labels'][i+3])
            }
            $('.label').on('click',function(){
                var ind = $(this).attr('id');
                $(".label").removeClass('selected');
                $(this).addClass('selected');
                $(".child").remove();
                add_rect("red",data['bounding_box'][ind]);
            });
            $('#0').click();         
            $("#waitbar").addClass('hidden');
            return false;
        },
        error: function (XMLHttpRequest, textStatus, errorThrown){
            console.log(XMLHttpRequest)
            $("#waitbar").addClass('hidden');
            alert("Elaboration Error!!");
            return false;

        }
    });
}


var add_rect = function(color, rect) {
    var $container = $("#container");
    $('<div class="child" />')
    .appendTo($container)
    .css("left", (rect['min_x']-15) + "px")
    .css("top", (rect['min_y']-15) + "px")
    .css("width", (rect['max_x']-rect['min_x']+30)+"px")
    .css("height", (rect['max_y']-rect['min_y']+30)+"px")
    .css("border", "5px solid " + color);

};

var add_label_info = function(index,label_text){
    label_text = label_text['label'];
    var $labels_container = $("#labels");
    $("<li><i class='fa fa-check' /><span class='label' id='"+index+"'>"+label_text+"</span></li>").appendTo($labels_container);
}   