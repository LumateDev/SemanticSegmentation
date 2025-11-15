<script setup>
</script>

<template>
  <el-container style="height: 100vh">
    <el-aside width="200px">
      <el-menu router>
        <el-menu-item index="/">Dashboard</el-menu-item>
        <el-menu-item index="/models">Модели</el-menu-item>
        <el-menu-item index="/train">Обучение</el-menu-item>
        <el-menu-item index="/predict">Предсказания</el-menu-item>
        <el-menu-item index="/stats">Статистика</el-menu-item>
      </el-menu>
    </el-aside>

    <el-main>
      <router-view />
    </el-main>
  </el-container>
</template>

<style scoped>
</style>