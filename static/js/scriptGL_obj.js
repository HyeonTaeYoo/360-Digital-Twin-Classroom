import { OBJLoader } from 'https://threejsfundamentals.org/threejs/resources/threejs/r127/examples/jsm/loaders/OBJLoader.js';
import { MTLLoader } from 'https://threejsfundamentals.org/threejs/resources/threejs/r127/examples/jsm/loaders/MTLLoader.js';
import { OrbitControls } from 'https://threejsfundamentals.org/threejs/resources/threejs/r127/examples/jsm/controls/OrbitControls.js';

const scene = new THREE.Scene(); // scene은 전역 변수로 고정
const objLoader = new OBJLoader(); // 오브젝트 파일 로더
const mtlLoader = new MTLLoader();
const loader = new THREE.TextureLoader(); // 텍스처 로더

function makeInstance(geometry, color, x) {
    const material = new THREE.MeshPhongMaterial({ color });

    const cube = new THREE.Mesh(geometry, material);
    scene.add(cube);
    cube.position.x = x;

    return cube;
}

function draw_line(color, x1, y1, z1, x2, y2, z2) {
    const material = new THREE.LineBasicMaterial({
        color: color
    });

    const points = [];
    points.push(new THREE.Vector3(x1, y1, z1));
    points.push(new THREE.Vector3(x2, y2, z2));

    const geometry = new THREE.BufferGeometry().setFromPoints(points);

    const line = new THREE.Line(geometry, material);
    scene.add(line);
}

function draw_axis() {
    draw_line(0xff0000, 10, 0, 0, 0, 0, 0); // x8-0
    draw_line(0x00ff00, 0, 10, 0, 0, 0, 0); // y
    draw_line(0x0000ff, 0, 0, 10, 0, 0, 0); // z
}

function resizeRendereToDisplaySize(renderer) {
    const canvas = renderer.domElement;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const needResize = canvas.width !== width || canvas.heigth !== height;

    if (needResize) {
        renderer.setSize(width, height, false);
    }
    return needResize;
}

function obj_load(path) {

    objLoader.load(path, (root) => {
        scene.add(root);
    });
}

function obj_load_with(path1, path2) {
    mtlLoader.load(path1, (mtl) => {
        mtl.preload();
        const objLoader = new OBJLoader();
        objLoader.setMaterials(mtl);
        objLoader.load(path2, (root) => {
            scene.add(root);
        });
    });
}



function make_classroom(renderer) {
    const texture = loader.load(

        'images/class_tex_front.jpg',
        () => {
            const rt = new THREE.WebGLCubeRenderTarget(texture.image.height);
            rt.fromEquirectangularTexture(renderer, texture);
            scene.background = rt.texture;
        });
}

function main() {
    // 내가 원하는 위치의 canvas에 project 띄우기
    const canvas = document.querySelector('#canvasID');

    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true });

    const fov = 80;
    const aspect = canvas.width / canvas.height;  // the canvas default
    const near = 0.1; // near와 far는 카메라로부터의 거리
    const far = 100;
    const camera = new THREE.PerspectiveCamera(fov, aspect, near, far);
    camera.position.x = -10
    const controls = new OrbitControls(camera, canvas);
    controls.target.set(0, 0, 0);
    controls.update();

    const boxWidth = 1;
    const boxHeight = 1;
    const boxDepth = 1;
    const geometry = new THREE.BoxGeometry(boxWidth, boxHeight, boxDepth);

    //const material = new THREE.MeshBasicMaterial({color: 0x44aa88});  // greenish blue
    const material = new THREE.MeshPhongMaterial({ color: 0x44aa88 }); // 광원에 반응하는 머터리얼

    // 광원 생성
    const color = 0xffffff; // 
    const intensity = 1.5;
    const amb_light = new THREE.AmbientLight(color, intensity);
    // 광원 위치 // target은 조명이 향하는 곳임 디폴트는 둘다 (0, 0,0)
    scene.add(amb_light);

    //draw_axis();

    make_classroom(renderer);

    obj_load_with('./material/Everyone.mtl', './obj/Everyone.obj');
    //obj_load_with('./material/JH.mtl', './obj/JH.obj');


    function render(time) {
        time *= 0.001;

        if (resizeRendereToDisplaySize(renderer)) {
            const canvas = renderer.domElement;
            camera.aspect = canvas.clientWidth / canvas.clientHeight; // 가로세로 비 갱신
            camera.updateProjectionMatrix(); // 투영 행렬 업데이트
        }

        renderer.render(scene, camera);
        requestAnimationFrame(render);
    }
    requestAnimationFrame(render); // 루프 시작
}

// 버튼 클릭 해당 부분
var btn1 = document.getElementById('submit1');
var btn2 = document.getElementById('submit2');

btn1.addEventListener('click', function () {
    //클릭시에 할 일
    console.log(btn1);
    btn1.style.color = 'red';
    btn2.style.color = 'black';
})

btn2.addEventListener('click', function () {
    //클릭시에 할 일
    console.log(btn2);
    btn2.style.color = 'blue';
    btn1.style.color = 'black';
})

main();