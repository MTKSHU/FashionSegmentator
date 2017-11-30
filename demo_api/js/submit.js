
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
    on_load(true);
    
    if($('#urls').val() == ""){
        formData.append('pic', $('input[type=file]')[0].files[0]); 
        formData.append('name', $('input[type=file]')[0].files[0].name.split('.')[0]);
    }
    else
        formData.append('name', $('#urls').val().split('/').pop().split('.')[0]);
    formData.append('urls', $('#urls').val());
    formData.append('zip_result', false);
    formData.append('heavy', true);
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

            var offs = 0;
            for(var j=0;j<data['labels'].length;j++){
                if(parseInt(data['labels'][j]['num']) < 3)
                    offs++;
                else{
                    add_label_info(j-offs,data['labels'][j]);  
                }
            }


            
            $('.label').on('click',function(){
                var ind = $(this).attr('id');
                $(".label").removeClass('selected');
                $(this).addClass('selected');
                $(".child").remove();
                add_rect("red",data['bounding_box'][ind]);
            });
            $('.label').first().click();         
            $("#waitbar").addClass('hidden');
            on_load(false);
            $("#clear_btn").click();
            return false;
        },
        error: function (XMLHttpRequest, textStatus, errorThrown){
            console.log(XMLHttpRequest)
            $("#waitbar").addClass('hidden');
            on_load(false);
            $("#clear_btn").click();
            alert("Elaboration Error!!");
            return false;

        }
    });
}


var add_rect = function(color, rect) {
    var $container = $("#container");
    $('<div class="child" />')
    .appendTo($container)
    .css("left", (rect['min_x']-5) + "px")
    .css("top", (rect['min_y']-5) + "px")
    .css("width", (rect['max_x']-rect['min_x']+10)+"px")
    .css("height", (rect['max_y']-rect['min_y']+10)+"px")
    .css("border", "3px solid " + color);

};

var add_label_info = function(index,label_text){
    label_text = label_text['label'];
    var $labels_container = $("#labels");
    $("<li><i class='fa fa-check' /><span class='label' id='"+index+"'>"+label_text+"</span></li>").appendTo($labels_container);
}   


var getObjects = function(obj, key, val) {
    var objects = [];
    for (var i in obj) {
        if (!obj.hasOwnProperty(i)) continue;
        if (typeof obj[i] == 'object') {
            objects = objects.concat(getObjects(obj[i], key, val));
        } else if (i == key && obj[key] == val) {
            objects.push(obj);
        }
    }
    return objects;
}


var max_dress = function(arr){
    max = 0;
    el_max = -1;
    for(var j=0;j<arr.length;j++)
        if(arr[j]!=undefined)
            if(arr[j]['score'] > max)
            {
                max = arr[j]['score'];
                el_max = arr[j];
            }
    return el_max;

}

var on_load  = function(load){
    
    $('input[type=file]').prop("disabled",load);
    $("#urls").prop("disabled",load);
    $("#clear_btn").prop("disabled",load);
    $("#send_data").prop("disabled",load);
    
    if(load)
        $(".main").css('opacity',0.05)
    else
        $(".main").css('opacity',1)
}