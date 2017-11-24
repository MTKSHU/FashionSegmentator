
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

            var rules_res = new Array(7);
            var label_rules = [5,7,11,13,15,16,17,18,19,20,21,22];
            var mapping_bb = new Array(25);
            for(var j=0;j<data['bounding_box'].length;j++){
                console.log(data['labels'][j+2]['num']);      
                if((label_rules.indexOf(parseInt(data['labels'][j+2]['num'])))!= -1) 
                    mapping_bb[data['labels'][j+2]['num']] = j;
                else
                    add_label_info(j,data['labels'][j+2]);  
            }


            var boots = getObjects(data,'num','5')[0];
            var dress = getObjects(data,'num','7')[0];
            var blazer = getObjects(data,'num','11')[0];
            var jeans = getObjects(data,'num','13')[0];
            var shirt = getObjects(data,'num','15')[0];
            var shoes = getObjects(data,'num','16')[0];
            var shorts = getObjects(data,'num','17')[0];
            var skirt = getObjects(data,'num','18')[0];
            var socks = getObjects(data,'num','19')[0];
            var cardigan = getObjects(data,'num','24')[0];
            var leggins = getObjects(data,'num','21')[0];
            var t_shirt = getObjects(data,'num','22')[0];
            var vest = getObjects(data,'num','23')[0];
            var vest2 = getObjects(data,'num','24')[0];

            var rule_el = max_dress([dress,skirt]);
            if(rule_el != -1){
                rules_res[0] = rule_el['label'];
                add_label_info(mapping_bb[rule_el['num']],rule_el);
            }

            var rule_el = max_dress([jeans,leggins,shorts]);
            if(rule_el != -1){
                rules_res[1] = rule_el['label'];
                add_label_info(mapping_bb[rule_el['num']],rule_el);
            }

            var rule_el = max_dress([boots,shoes,socks]);
            if(rule_el != -1){
                rules_res[2] = rule_el['label'];
                add_label_info(mapping_bb[rule_el['num']],rule_el);
            }

            var rule_el = max_dress([vest,vest2]);
            if(rule_el != -1){
                rules_res[3] = rule_el['label'];
                add_label_info(mapping_bb[rule_el['num']],rule_el);
            }

            var rule_el = max_dress([shirt,t_shirt,blazer]);
            if(rule_el != -1){
                rules_res[4] = rule_el['label'];
                add_label_info(mapping_bb[rule_el['num']],rule_el);
            }
            
            
            if(rules_res[0] == undefined && rules_res[4] == undefined ){
                var rule_el = max_dress([shirt,t_shirt,dress]);
                if(rule_el != -1){
                    rules_res[5] = rule_el['label'];
                    add_label_info(mapping_bb[rule_el['num']],rule_el);
                }
            }
            
            if(rules_res[4] != 'jacket/blazer') { 
                var rule_el = max_dress([cardigan,blazer]);
                if(rule_el != -1){
                    rules_res[6] = rule_el['label'];
                    add_label_info(mapping_bb[rule_el['num']],rule_el);
                }
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