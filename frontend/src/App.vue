<template>
  <div id="app">
    <AppHeader />
    <main class="main-content">
      <RouterView />
    </main>
  </div>
</template>

<script setup lang="ts">
import AppHeader from '@/components/AppHeader.vue';
</script>

<style scoped>
#app {
  height: 100vh; /* ВАЖНО: вместо 100% */
  display: flex;
  flex-direction: column;
  overflow: hidden; /* ОСТАВИТЬ: блокируем скролл страницы */
}

.main-content {
  flex: 1;
  overflow: hidden; /* ОСТАВИТЬ: блокируем скролл страницы */
}
</style>
