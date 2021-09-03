function main() {
    // 내가 원하는 위치의 canvas에 project 띄우기
    const canvas = document.querySelector('#canvasID');
    const renderer = new THREE.WebGLRenderer({canvas});
    //  건들지 말기
    
    const fov = 75;
    const aspect = 2;  // the canvas default
    const near = 0.1;
    const far = 5;
    const camera = new THREE.PerspectiveCamera(fov, aspect, near, far);
    camera.position.z = 2;

    const scene = new THREE.Scene();

    const boxWidth = 1;
    const boxHeight = 1;
    const boxDepth = 1;
    const geometry = new THREE.BoxGeometry(boxWidth, boxHeight, boxDepth);

    const material = new THREE.MeshBasicMaterial({color: 0x44aa88});  // greenish blue

    const cube = new THREE.Mesh(geometry, material);
    scene.add(cube);

    // # 아래부분이 추가되었음
    function render(time) {
        time *= 0.001;  // convert time to seconds
        cube.rotation.x = time;
        cube.rotation.y = time;

        renderer.render(scene, camera);
        requestAnimationFrame(render);
    }
    requestAnimationFrame(render);
}

// 버튼 클릭 해당 부분
var btn = document.getElementById('submit1');
btn.addEventListener('click', function(){
    //클릭시에 할 일
    btn.style.color = 'red';
})

var btn = document.getElementById('submit2');
btn.addEventListener('click', function(){
    //클릭시에 할 일
    btn.style.color = 'red';
})

main();