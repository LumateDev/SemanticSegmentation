import { createRouter, createWebHistory } from 'vue-router';
import HomePage from '@/pages/home-page/HomePage.vue';

const routes = [
  { path: '/', component: HomePage },
  { path: '/home', component: HomePage },
  { path: '/models', component: () => import('@/pages/models-page/ModelsPage.vue') },
  { path: '/datasets', component: () => import('@/pages/datasets-page/DatasetsPage.vue') },
  { path: '/test', component: () => import('@/pages/test-page/TestPage.vue') },
  { path: '/train', component: () => import('@/pages/train-page/TrainPage.vue') },
  { path: '/predict', component: () => import('@/pages/predict-page/PredictPage.vue') },
  { path: '/compare', component: () => import('@/pages/compare-page/ComparePage.vue') },
];

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes,
});

export default router;
