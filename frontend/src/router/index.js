import { createRouter, createWebHistory } from 'vue-router'

import Dashboard from '../pages/Dashboard.vue'
import Models from '../pages/Models.vue'
import Train from '../pages/Train.vue'
import Predict from '../pages/Predict.vue'
import Stats from '../pages/Stats.vue'

const routes = [
  { path: '/', component: Dashboard },
  { path: '/models', component: Models },
  { path: '/train', component: Train },
  { path: '/predict', component: Predict },
  { path: '/stats', component: Stats }
]

export const router = createRouter({
  history: createWebHistory(),
  routes,
})