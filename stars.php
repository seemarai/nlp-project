<html>
<style type="text/css">
	.starRate {position:relative; margin:20px; overflow:hidden; zoom:1;}
	.starRate ul {width:160px; margin:0; padding:0;}
	.starRate li {display:inline; list-style:none;}
	.starRate a, .starRate b {background:url(star_rate.gif) left top repeat-x;}
	.starRate a {float:right; margin:0 80px 0 -144px; width:80px; height:16px; background-position:left 16px; color:#000; text-decoration:none;}
	.starRate a:hover {background-position:left -32px;}
	.starRate b {position:absolute; z-index:-1; width:80px; height:16px; background-position:left -16px;}
	.starRate div b {left:0px; bottom:0px; background-position:left top;}
	.starRate a span {position:absolute; left:-300px;}
	.starRate a:hover span {left:90px; width:100%;}
</style>
<body>
<div class="starRate">
	<div>Currently rated: <?php if(isset($_GET["pos"])) echo $_GET["pos"]; else echo 0;?> stars<b></b></div>
		<ul>
			<li><a href="/stars.php?pos=5"><span>Give it 5 stars</span><?php if($_GET["pos"]==5) echo "<b></b>";?></a></li>
			<li><a href="/stars.php?pos=4"><span>Give it 4 stars</span><?php if($_GET["pos"]==4) echo "<b></b>";?></a></li>
			<li><a href="/stars.php?pos=3"><span>Give it 3 stars</span><?php if($_GET["pos"]==3) echo "<b></b>";?></a></li>
			<li><a href="/stars.php?pos=2"><span>Give it 2 stars</span><?php if($_GET["pos"]==2) echo "<b></b>";?></a></li>
			<li><a href="/stars.php?pos=1"><span>Give it 1 star</span><?php if($_GET["pos"]==1) echo "<b></b>";?></a></li>
		</ul>
	</div>
</body>
</html>