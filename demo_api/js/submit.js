var scale = 0;
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

    $(".alert").remove();
    var formData = new FormData(); 
    // Main magic with files here 
    var base_url = "http://localhost:8000/"; 
     
    $("#waitbar").removeClass('hidden'); 
    on_load(true); 
     
    if($('#urls').val() == "" ){
        formData.append('pic', $('input[type=file]')[0].files[0]);  
        formData.append('name', $('input[type=file]')[0].files[0].name.split('.')[0]); 
    } 
    else 
        formData.append('name', $('#urls').val().split('/').pop().split('.')[0]); 
    formData.append('urls', $('#urls').val()); 

    formData.append('zip_result', false); 
    formData.append('heavy', true); 
    $("#clear_btn").click(); 
    var result = 
        $.ajax({
        url: base_url + 'images/',
        type:"POST",
        data: formData,     
        contentType: false,
        processData:false,
        dataType:"json",
        cache:false,
        success: function(data){
            //http://localhost:8000/media/pics/""
            //console.log(data);
            $("#box_result").prop("hidden",false);
            var pic_path = base_url + data['pic'].split('/')[5] + '/' + data['pic'].split('/')[6] + '/' +data['pic'].split('/')[7]; 
            $("#res_img").attr('src',pic_path);
			$("#res_img").attr('style','max-width:600;max-height:600;');
			scale = 1;
            //$("li").remove();
            $(".child_circ").remove();
            $(".child").remove();
            var offs = 0;
            for(var j=0;j<data['labels'].length;j++){
                if(parseInt(data['labels'][j]['num']) < 3)
                    offs++;
                else{
                    add_circ('#0088cc',j-offs,data['bounding_box'][j-offs]);  
                }
            }


            /*
            $('.label').on('click',function(){
                var ind = parseInt($(this).attr('id'));
                $(".label").removeClass('selected');
                $(this).addClass('selected');
                $(".child").remove();
                add_rect("#0088cc",data['bounding_box'][ind],data['labels'][ind+offs]['label']);
                $(".child_circ").prop("hidden",true);
            });
        */

            $(".child_circ").on('click',function(){
                var ind = parseInt($(this).attr('id'));
                $(".child").remove();
                add_rect("#0088cc",data['bounding_box'][ind],data['labels'][ind+offs]['label']);
                $(".child_circ").prop("hidden",false);
                $(this).prop("hidden",true);
            });
            $("#waitbar").addClass('hidden');
            on_load(false);
            return false;
        },
        error: function (XMLHttpRequest, textStatus, errorThrown){
            console.log(XMLHttpRequest)
            $("#waitbar").addClass('hidden');
            on_load(false);
            if(XMLHttpRequest.responseText.indexOf("Impossibile") != -1)
            
                $("#container").append('<div class="alert alert-warning" role="alert"><i class="fa fa-exclamation-triangle" aria-hidden="true"></i>Errore! L\'indirizzo dell\'immagine fornito non è corretto!</div>')
                //alert("Errore! L'indirizzo dell'immagine fornito non è corretto!")
            else $("#container").append('<div class="alert alert-warning" role="alert"><i class="fa fa-exclamation-triangle" aria-hidden="true"></i>'+errorThrown+'</div>')
            return false;

        }
    });

}


var add_rect = function(color, rect,text) {
    var $container = $("#container");
    $('<span class="label label-primary">'+text+'</span>').appendTo(
        $('<div class="child" />')
        .appendTo($container)
        .css("left", ((rect['min_x']/scale)-5) + "px")
        .css("top", (rect['min_y']/scale-5) + "px")
        .css("width", (rect['max_x']/scale-rect['min_x']/scale+10)+"px")
        .css("height", (rect['max_y']/scale-rect['min_y']/scale+10)+"px")
        .css("border", "3px solid " + color))
    .css("position","relative")
    .css("left", "-3px")
    .css("top", "-7px");



};
var add_circ = function(color,index, rect){
    var $container = $("#container");
    $('<div class="child_circ" id="'+index+'"/>')
    .appendTo($container)
    .css("left", ((rect['min_x']/scale+ rect['max_x']/scale)/2)+ "px")
    .css("top", ((rect['min_y']/scale+ rect['max_y']/scale)/2) + "px")
    .css("background-color", color)
    .css("cursor","pointer")
    .css("border", "3px solid #fff");
}


var on_load  = function(load){
    
    $('input[type=file]').prop("disabled",load);
    $("#urls").prop("disabled",load);
    $("#send_data").prop("disabled",load);
    
    if(load)
        $(".main").css('opacity',0.05)
    else{
        $(".main").css('opacity',1)
    }
}

