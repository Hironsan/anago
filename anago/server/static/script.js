function getMarkupText(val, data){
    var res = [];
    var dic = {"PER": "person", "LOC": "location", "ORG": "organization", "MISC": "misc"};

    for (key in data) {
        for (var i = 0; i < data[key].length; i++) {
            var w = data[key][i];
            var text = '<small class="axlabel ' + dic[key] + '">' + w + '</small>';
            var regExp = new RegExp(w, "g" ) ;
            val = val.replace(regExp, text)
        }
    }
    return val;
}

$(function () {
	$("#sentence").keypress(function (e) {
		var code = (e.keyCode ? e.keyCode : e.which);
		if (code == 13) {
		    var val = $("textarea#sentence").val();
		    console.log("value"+val);
		    $.post("/", {"sent": val},
			   function(data, status){
			       console.log(data);
			       var text = getMarkupText(val, JSON.parse(data));
			       $(".message-body").html(text);
			       console.log(text);
			});
		    //$("#submit").trigger('click');
		    return true;
		}
	    });
    });